import argparse
import operator
import pathlib
import shutil

import numpy as np
import taichi as ti
import cgraph as cgr

parser = argparse.ArgumentParser()
parser.add_argument('--exp',
                    choices=['implicit', 'explicit'],
                    default='implicit')
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--aot', default=False, action='store_true')
args = parser.parse_args()

ti.init(arch=ti.vulkan, dynamic_index=False, packed=True)

E, nu = 5e4, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # lambda = 0
density = 1000.0
dt = 2e-4

if args.exp == 'implicit':
    dt = 1e-2

n_cube = np.array([8] * 3)
n_verts = np.product(n_cube)
n_cells = 5 * np.product(n_cube - 1)
dx = 1 / (n_cube.max() - 1)

print(f'n_cube={n_cube} n_verts={n_verts} n_cells={n_cells} dx={dx}')


def compute_indices_shape():
    su = 0
    for i in range(3):
        su += (n_cube[i] - 1) * (n_cube[(i + 1) % 3] - 1)
    return 2 * su * 2 * 3


# Taichi fields
# x = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
x = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
ox = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
v = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
f = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
mul_ans = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
m = ti.field(dtype=ti.f32, shape=n_verts)

vertices = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

B = ti.Matrix.field(args.dim, args.dim, dtype=ti.f32, shape=n_cells)
W = ti.field(dtype=ti.f32, shape=n_cells)

b = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
r0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
p0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

indices = ti.field(ti.i32, shape=compute_indices_shape())


@ti.func
def i2p(I):
    return (I.x * n_cube[1] + I.y) * n_cube[2] + I.z


@ti.func
def set_element(e, I, verts):
    for i in ti.static(range(args.dim + 1)):
        vertices[e][i] = i2p(I + (([verts[i] >> k for k in range(3)] ^ I) & 1))


@ti.kernel
def get_vertices():
    '''
    This kernel partitions the cube into tetrahedrons.
    Each unit cube is divided into 5 tetrahedrons.
    '''
    for I in ti.grouped(ti.ndrange(*(n_cube - 1))):
        e = ((I.x * (n_cube[1] - 1) + I.y) * (n_cube[2] - 1) + I.z) * 5
        for i, j in ti.static(enumerate([0, 3, 5, 6])):
            set_element(e + i, I, (j, j ^ 1, j ^ 2, j ^ 4))
        set_element(e + 4, I, (1, 2, 4, 7))
    for I in ti.grouped(ti.ndrange(*(n_cube))):
        ox[i2p(I)] = I * dx


@ti.func
def Ds(verts, x:ti.template()):
    return ti.Matrix.cols([x[verts[i]] - x[verts[3]] for i in range(3)])


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def get_force_func(c, verts, x: ti.template()):
    F = Ds(verts, x) @ B[c]
    P = ti.Matrix.zero(ti.f32, 3, 3)
    U, sig, V = ssvd(F)
    P = 2 * mu * (F - U @ V.transpose())
    H = -W[c] * P @ B[c].transpose()
    for i in ti.static(range(3)):
        force = ti.Vector([H[j, i] for j in range(3)])
        f[verts[i]] += force
        f[verts[3]] -= force


@ti.kernel
def get_force(x: ti.any_arr()):
    for c in vertices:
        get_force_func(c, vertices[c], x)
    for u in f:
        f[u].y -= 9.8 * m[u]


@ti.kernel
def matmul_cell(ret: ti.template(), vel: ti.template()):
    for i in ret:
        ret[i] = vel[i] * m[i]
    for c in vertices:
        verts = vertices[c]
        W_c = W[c]
        B_c = B[c]
        for u in ti.static(range(4)):
            for d in ti.static(range(3)):
                dD = ti.Matrix.zero(ti.f32, 3, 3)
                if ti.static(u == 3):
                    for j in ti.static(range(3)):
                        dD[d, j] = -1
                else:
                    dD[d, u] = 1
                dF = dD @ B_c
                dP = 2.0 * mu * dF
                dH = -W_c * dP @ B_c.transpose()
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        tmp = (vel[verts[i]][j] - vel[verts[3]][j])
                        ret[verts[u]][d] += -dt**2 * dH[j, i] * tmp


@ti.kernel
def add(ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
    for i in ans:
        ans[i] = a[i] + k * b[i]


@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> ti.f32:
    ans = 0.0
    for i in a:
        ans += a[i].dot(b[i])
    return ans


@ti.kernel
def get_b():
    for i in b:
        b[i] = m[i] * v[i] + dt * f[i]


def cg():

    def mul(x):
        matmul_cell(mul_ans, x)
        return mul_ans

    get_force()
    get_b()
    add(r0, b, -1, mul(v))

    d = p0
    d.copy_from(r0)
    r_2 = dot(r0, r0)
    n_iter = 8
    epsilon = 1e-6
    r_2_init = r_2
    r_2_new = r_2
    for _ in range(n_iter):
        q = mul(d)
        alpha = r_2_new / dot(d, q)
        add(v, v, alpha, d)
        add(r0, r0, -alpha, q)
        r_2 = r_2_new
        r_2_new = dot(r0, r0)
        if r_2_new <= r_2_init * epsilon**2: break
        beta = r_2_new / r_2
        add(d, r0, beta, d)
    f.fill(0)
    add(x, x, dt, v)


def cg_cgraphed():
    def make_mul_fn():
        param_x = cgr.Var('x')
        lit_mul_ans = cgr.Literal(mul_ans)
        fn_bb = cgr.empty_basic_block()
        fn_bb.append(
            cgr.NativeFunc(matmul_cell).invoke_with(lit_mul_ans,
                                                    param_x).as_stmt())
        fn_bb.append(cgr.Return(lit_mul_ans))
        return cgr.FuncDef('mul', [param_x], fn_bb)

    bb = cgr.empty_basic_block()
    df_mul = make_mul_fn()
    df_mul_cb = df_mul.callable
    bb.append(df_mul)
    bb.append(cgr.NativeFunc(get_force).invoke().as_stmt())
    bb.append(cgr.NativeFunc(get_b).invoke().as_stmt())
    lit_v = cgr.Literal(v)
    mul_v_expr = df_mul_cb.invoke_with(lit_v)
    lit_r0 = cgr.Literal(r0)
    nfn_add = cgr.NativeFunc(add)
    bb.append(
        nfn_add.invoke_with(lit_r0, cgr.Literal(b), cgr.Literal(-1),
                            mul_v_expr).as_stmt())
    bb.append(cgr.NativeFunc(p0.copy_from).invoke_with(lit_r0).as_stmt())
    var_d = cgr.Var('d')
    bb.append(cgr.Assign(var_d, cgr.Literal(p0)))
    var_r_2 = cgr.Var('r_2')
    nfn_dot = cgr.NativeFunc(dot)
    bb.append(cgr.Assign(var_r_2, nfn_dot.invoke_with(lit_r0, lit_r0)))
    var_r_2_init = cgr.Var('r_2_init')
    var_r_2_new = cgr.Var('r_2_new')
    var_iter = cgr.Var('iter')
    bb.append(cgr.Assign(var_r_2_init, var_r_2))
    bb.append(cgr.Assign(var_r_2_new, var_r_2))
    bb.append(cgr.Assign(var_iter, cgr.Literal(0)))

    def make_iterative():
        loop_bb = cgr.empty_basic_block()
        loop_bb.append(cgr.make_self_inc(var_iter))
        var_q = cgr.Var('q')
        loop_bb.append(cgr.Assign(var_q, df_mul_cb.invoke_with(var_d)))
        var_alpha = cgr.Var('alpha')
        div_expr = cgr.NativeFunc(lambda x, y: x / y).invoke_with(
            var_r_2_new, nfn_dot.invoke_with(var_d, var_q))
        loop_bb.append(cgr.Assign(var_alpha, div_expr))
        loop_bb.append(
            nfn_add.invoke_with(lit_v, lit_v, var_alpha, var_d).as_stmt())
        loop_bb.append(
            nfn_add.invoke_with(lit_r0, lit_r0, cgr.Neg(var_alpha),
                                var_q).as_stmt())
        loop_bb.append(cgr.Assign(var_r_2, var_r_2_new))
        loop_bb.append(
            cgr.Assign(var_r_2_new, nfn_dot.invoke_with(lit_r0, lit_r0)))
        lit_eps2 = cgr.Literal((1e-6)**2)
        if_cond = cgr.NativeFunc(operator.le).invoke_with(
            var_r_2_new,
            cgr.NativeFunc(operator.mul).invoke_with(var_r_2_init, lit_eps2))
        if_t_bb = cgr.empty_basic_block()
        if_t_bb.append(cgr.Break())
        loop_bb.append(cgr.If(if_cond, if_t_bb))
        var_beta = cgr.Var('beta')
        loop_bb.append(
            cgr.Assign(
                var_beta,
                cgr.NativeFunc(operator.truediv).invoke_with(
                    var_r_2_new, var_r_2)))
        loop_bb.append(
            nfn_add.invoke_with(var_d, lit_r0, var_beta, var_d).as_stmt())
        cond = cgr.NativeFunc(operator.lt).invoke_with(var_iter,
                                                       cgr.Literal(50))

        return cgr.While(cond, loop_bb)

    bb.append(make_iterative())
    bb.append(cgr.NativeFunc(f.fill).invoke_with(cgr.Literal(0)).as_stmt())
    lit_x = cgr.Literal(x)
    bb.append(
        nfn_add.invoke_with(lit_x, lit_x, cgr.Literal(dt), lit_v).as_stmt())

    vm = cgr.VM()
    vm.exec(bb)

    return bb


@ti.kernel
def advect(x: ti.any_arr()):
    for p in x:
        v[p] += dt * (f[p] / m[p])
        # x[p] -= dt * 0.05
        x[p] += dt * v[p]
        f[p] = ti.Vector([0, 0, 0])


@ti.kernel
def init(x: ti.any_arr()):
    for u in x:
        x[u] = ox[u]
        v[u] = [0.0] * 3
        f[u] = [0.0] * 3
        m[u] = 0.0
    for c in vertices:
        F = Ds(vertices[c], x)
        B[c] = F.inverse()
        W[c] = ti.abs(F.determinant()) / 6
        for i in ti.static(range(4)):
            m[vertices[c][i]] += W[c] / 4 * density
    for u in x:
        x[u].y += 1.0


@ti.kernel
def floor_bound(x: ti.any_arr()):
    for u in x:
        if x[u].y < 0:
            x[u].y = 0
            if v[u].y < 0:
                v[u].y = 0


@ti.func
def check(u):
    ans = 0
    rest = u
    for i in ti.static(range(3)):
        k = rest % n_cube[2 - i]
        rest = rest // n_cube[2 - i]
        if k == 0: ans |= (1 << (i * 2))
        if k == n_cube[2 - i] - 1: ans |= (1 << (i * 2 + 1))
    return ans


@ti.kernel
def get_indices(x: ti.any_arr()):
    # calculate all the meshes on surface
    cnt = 0
    for c in vertices:
        if c % 5 != 4:
            for i in ti.static([0, 2, 3]):
                verts = [vertices[c][(i + j) % 4] for j in range(3)]
                sum = check(verts[0]) & check(verts[1]) & check(verts[2])
                if sum:
                    m = ti.atomic_add(cnt, 1)
                    det = ti.Matrix.rows([
                        x[verts[i]] - [0.5, 1.5, 0.5] for i in range(3)
                    ]).determinant()
                    if det < 0:
                        tmp = verts[1]
                        verts[1] = verts[2]
                        verts[2] = tmp
                    indices[m * 3] = verts[0]
                    indices[m * 3 + 1] = verts[1]
                    indices[m * 3 + 2] = verts[2]

_printed_ast = False
def substep():
    global _printed_ast
    if args.exp == 'explicit':
        for _ in range(40):
            get_force(x)
            advect(x)
            # print(f'after one substep:\n{x.to_numpy()}\n')
    else:
        # cg()
        bb = cg_cgraphed()
        if not _printed_ast:
          vis = cgr.Printer()
          vis.visit(bb)
          print(vis.dump())
          _printed_ast = True
    floor_bound(x)


def run_sim():
    get_vertices()
    init(x)
    get_indices(x)
    print(f'init:{x.shape}\n{x.to_numpy()}\n')

    def T(a):

        phi, theta = np.radians(28), np.radians(32)

        a = a - 0.2
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        c, s = np.cos(phi), np.sin(phi)
        C, S = np.cos(theta), np.sin(theta)
        x, z = x * c + z * s, z * c - x * s
        u, v = x, y * C + z * S
        return np.array([u, v]).swapaxes(0, 1) + 0.5

    gui = ti.GUI('Implicit FEM')
    frame = 0
    while gui.running:
        substep()
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
        if gui.is_pressed('r'):
            init(x)
        gui.clear(0x000000)
        xnp = x.to_numpy()
        xpos = T(xnp / 3)
        # if frame % 50 == 0:
        #   print(f'frame={frame}')
        #   for i in range(n_verts):
        #     print(f'  i={i} {xnp[i, :]}')
        gui.circles(xpos, radius=1.5, color=0xba543a)
        gui.show()
        frame += 1


def run_aot_explicit(m):
    m.add_kernel(get_force, (x,))
    m.add_kernel(advect, (x,))


def run_aot_implicit(m):
    with m.add_kernel_template(matmul_cell) as kt:
        kt.instantiate(ret=mul_ans, vel=x)
        kt.instantiate(ret=mul_ans, vel=x)


def run_aot():
    dir_name = args.exp + '_fem'
    shutil.rmtree(dir_name, ignore_errors=True)
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=False)

    m = ti.aot.Module(ti.metal)
    m.add_kernel(get_vertices)
    m.add_kernel(init, (x,))
    m.add_kernel(get_indices, (x, ))
    m.add_kernel(floor_bound, (x,))
    if args.exp == 'explicit':
        run_aot_explicit(m)
    else:
        run_aot_implicit(m)

    # m.add_field('x', x)

    m.save(dir_name, 'fem')
    print('AOT done')


if __name__ == '__main__':
    if args.aot:
        run_aot()
    else:
        run_sim()