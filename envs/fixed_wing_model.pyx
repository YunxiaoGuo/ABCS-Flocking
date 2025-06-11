# cython: language_level=3
# cython: simd=True
cimport cython
import numpy as np
cimport numpy as npc
from cython.parallel import prange
from numpy.random import default_rng
from libc.math cimport sqrt,pow,M_PI,cos,sin,tan,exp,isnan


# basic functions: implement the fundamental operations
cdef object rng = default_rng()

@cython.boundscheck(False)
@cython.wraparound(False)
def double_array_to_list(double[:, ::1] arr):
    # TODO: convert double[:, ::1] to Python list
    cdef Py_ssize_t i, j
    cdef Py_ssize_t nrows = arr.shape[0]
    cdef Py_ssize_t ncols = arr.shape[1]
    result = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            if isnan(arr[i, j]):
                row.append(float('nan'))
            else:
                row.append(arr[i, j])
        result.append(row)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def nparray_to_double_array(object arr):
    # TODO: convert numpy.ndarray to double[:,::1]
    cdef npc.ndarray[npc.double_t, ndim=2] np_arr
    try:
        np_arr = np.ascontiguousarray(arr, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError("can not convert numpy.ndarray to double[:,::1]") from e
    return np_arr

@cython.boundscheck(False)
@cython.wraparound(False)
def double_array_to_nparray(double[:,::1] arr):
    # TODO: convert double[:,::1] to numpy.ndarray
    cdef npc.ndarray[npc.double_t, ndim = 2] np_arr = np.asarray(arr)
    return np_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float generate_randoms():
    # TODO: generate random number with high performance rng
    return rng.random()


cdef inline double int_to_double(long x) nogil:
    # TODO: convert int to float
    return <double> x

cdef inline double float32_to_float64(float x) nogil:
    # TODO: convert float32 to float64
    return <double>x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double clip(double x,
         double min_val,
         double max_val):
    # TODO: Clip function
    cdef double result
    if min_val > max_val:
        raise ValueError("min_val should not larger than max_val")
    if x < min_val:
        result = min_val
    elif x > max_val:
        result = max_val
    else:
        result = x
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[::1] matvec_mult(double[:, ::1] A, int m, int n, double[::1] x):
    # TODO: matrix and vector multiplication
    cdef:
        double[::1] b = np.zeros(m, dtype=np.float64)
        int i, j
    for i in prange(m,nogil=True):
        for j in range(n):
            b[i] += A[i, j] * x[j]
    return np.asarray(b)

cdef double cabs(double x):
    cdef double y
    if x > 0:
        y = x
    else:
        y = -x
    return y

# reward function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double get_F(double[:] state_i, double[:] state_j, double Cv, double sigma, double beta, double Cr):
    # TODO: get F, core of the Leader-Guided Cucker-Smale Flocking Reward
    cdef:
        double x_i = state_i[0], y_i = state_i[1], theta_i = state_i[2]
        double v_i = cabs(state_i[4])
        double x_j = state_j[0], y_j = state_j[1], theta_j = state_j[2]
        double v_j = cabs(state_j[4])
        double dx = x_i - x_j
        double dy = y_i - y_j
        double cos_dtheta = cos(theta_i - theta_j)
        double delta_v_sq = v_i * v_i - 2 * v_i * v_j * cos_dtheta + v_j * v_j
        double delta_r_sq = dx * dx + dy * dy
        double denom_base = sigma * sigma + delta_r_sq
        double temp
    if delta_v_sq > 0:
        temp = sqrt(delta_v_sq) + Cv
    else:
        temp = Cv
    return temp / pow(denom_base, beta) + Cr * sqrt(delta_r_sq)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_r_sup(double Cv, double Cr, double sigma, double beta,double v_max):
    cdef double r_sup, temp, episode_time = 100
    temp = pow(sigma, 2 * beta)
    r_sup = (Cv + 2 * v_max * (1 + temp * Cr * episode_time)) / temp
    return r_sup * r_sup

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double L_G_CSreward(double[:,::1] state, int num, int n, double Cv, double sigma, double beta, double Cr, double theta):
    # TODO: Leader-guided Cucker-Smale Flocking Reward
    cdef:
        double F , reward
        double[:] alpha = np.empty(n+1, dtype=np.float64)
        int i
        double temp, sms = 0.0, f= 0.0
    for i in range(n+1):
        F = get_F(state[num], state[i], Cv, sigma, beta, Cr)
        alpha[i] = get_F(state[i], state[0], Cv, sigma, beta, Cr) if i != 0 else 0.0
        temp = exp(-theta * alpha[i])
        f += temp*F
        sms += temp
    f /= sms
    reward = - f * f
    return reward

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double M_CSreward(double[:,::1] state, int num, int n, double Cv, double sigma, double beta, double Cr, double theta):
    # TODO: Modified Cucker-Smale Flocking Reward
    cdef:
        double F , reward,f= 0.0
        int i
    for i in range(n+1):
        F = get_F(state[num], state[i], Cv, sigma, beta, Cr)
        f += F
    reward = - f * f
    return reward

cdef double CSreward(double[:,::1] state, int num, int n, double sigma, double beta):
    # TODO: Traditional Cucker-Smale Flocking Reward
    cdef double reward = 0
    cdef int i
    for i in range(n+1):
        reward += sqrt(state[num, 4] * state[num, 4] - 2 * cabs(state[num, 4]) * cabs(state[0, 4]) * cos(
            state[num, 2] - state[0, 2]) + (state[0, 4]) ** 2) / sqrt(
            sigma * sigma + pow((state[num, 0] - state[0, 0]) ** 2 + (state[num, 1] - state[0, 1]) ** 2, beta))
    reward = reward/n
    return - reward * reward

cdef double Q_flocking_reward(double[:,::1] state, int num):
    cdef:
        double d1 = 40
        double d2 = 65
        double omega = 0.05
        double rho,delta_x,delta_y,delta_psi,d,reward
    delta_x = state[0,0] - state[num,0]
    delta_y = state[0,1] - state[num,1]
    delta_psi = state[num,2] - state[0,2]
    rho = sqrt(delta_x * delta_x + delta_y * delta_y)
    d = d1 - rho if d1 - rho > rho - d2 else rho - d2
    d = d if d > 0.0 else 0.0
    reward = d1 * delta_psi / (M_PI * (1 + omega * d))
    reward = reward if reward > d else d
    return - reward

# functional functions: implement the fixed-wing UAV kinematic model

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,::1] generator(
        double x_grid,
        double y_grid,
        double delta_r_0,
        int n,
):
    # TODO: initialize the fixed-wing UAV flocking
    cdef:
        double[:,::1] state = np.empty((n+1,5), dtype=np.float64)
        double[:] x=np.empty(n+1, dtype=np.float64)
        double[:] y=np.empty(n+1, dtype=np.float64)
        double[:] v=np.empty(n+1, dtype=np.float64)
        double[:] psi=np.empty(n+1, dtype=np.float64)
        double[:] roll=np.empty(n+1, dtype=np.float64)
        int k = 0, i, j, m = int(sqrt(n))
    for i in range(m + 1):
        for j in range(m + 1):
            x_grid += i * delta_r_0
            y_grid += j * delta_r_0
            x[k] = x_grid + generate_randoms()*5
            y[k] = y_grid + generate_randoms()*5
            psi[k] = generate_randoms()*M_PI
            roll[k] = generate_randoms()*M_PI
            v[k] = 12 + generate_randoms()
            k += 1
            if k == n + 1:
                break
        if k == n + 1:
            break
    for j in prange(n+1,nogil=True):
        state[j,0]=x[j]
        state[j,1]=y[j]
        state[j,2]=psi[j]
        state[j,3]=roll[j]
        state[j,4]=v[j]
    return state

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,::1] rot_matrix(double psi_0):
    cdef:
        double[:,::1] rot = np.empty((2,2),dtype=np.float64)
        double cos_psi_l = cos(psi_0)
        double sin_psi_l = sin(psi_0)
    rot[0, 0] = cos_psi_l
    rot[0, 1] = sin_psi_l
    rot[1, 0] = -sin_psi_l
    rot[1, 1] = cos_psi_l
    return rot

@cython.boundscheck(False)
@cython.wraparound(False)
def reset(
        double x_grid,
        double y_grid,
        double delta_r_0,
        int n
):
    # TODO: reset for the RL environment
    cdef:
        double[:,::1] state = generator(x_grid, y_grid, delta_r_0, n)
        double[:,::1] rstate_ = Convert(state, n)
        npc.ndarray[npc.double_t, ndim = 2] rstate = np.asarray(rstate_)
    return state, rstate

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] Convert(double[:,::1] state, int n):
    # TODO: Covert UAVs' state to leader-relative state
    cdef:
        double[::1] vector=np.empty(2,dtype=np.float64)
        double[::1] s12=np.empty(2,dtype=np.float64)
        double[:,::1] rstate = np.empty((n,7),dtype=np.float64)
        double psi_0 = state[0,2]
        double[:, ::1] rot = rot_matrix(psi_0)
        double x_0 = state[0,0], y_0 = state[0,1], roll_0 = state[0,3], v_0 = state[0,4]
        double x_i, y_i, psi_i, roll_i, v_i, delta_psi_i
        int i
    for i in range(1, n+1):
        '''
        rotation matrix of leader
        R = [ cos(psi_l),sin(psi_l)  ]
            [ -sin(psi_l),cos(psi_f) ]
        '''
        x_i = state[i,0]
        y_i = state[i,1]
        psi_i = state[i,2]
        roll_i = state[i,3]
        v_i = state[i,4]
        delta_psi_i = psi_i - psi_0
        vector[0] = x_i - x_0
        vector[1] = state[i,1] - y_0
        s12 = matvec_mult(rot, 2, 2, vector)
        rstate[i - 1, 0] = s12[0]
        rstate[i - 1, 1] = s12[1]
        rstate[i - 1, 2] = delta_psi_i
        rstate[i - 1, 3] = roll_0
        rstate[i - 1, 4] = roll_i
        rstate[i - 1, 5] = v_0
        rstate[i - 1, 6] = v_i
    return rstate



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] crash_jud(double[:,::1] state, int n, double wing_length, double air_length):
    # TODO: Collision Judgement
    cdef:
        double coll_dis, delta_x, delta_y, delta_r
        double[:] done = np.empty(n,dtype=np.float64)
        double[:,::1] dis = np.zeros((n+1,n+1), dtype=np.float64)
        int i,j,k = 0
    coll_dis = wing_length if wing_length > air_length else air_length
    for i in range(n+1):
        for j in range(i, n+1):
            if i == j:
                dis[i, j] = 0
            else:
                delta_x = state[i,0] - state[j,0]
                delta_y = state[i,1] - state[j,1]
                delta_r = sqrt(delta_x * delta_x + delta_y * delta_y)
                dis[i, j] = 0 if delta_r > coll_dis else 1
    done = np.sum(dis,axis=1)
    return done[1:]

@cython.boundscheck(False)
@cython.wraparound(False)
def step(
        double[:,::1] state,
        int n,
        npc.ndarray[npc.double_t, ndim = 2] actions_,
        double rollmax,
        double vmax,
        double wing_length,
        double air_length,
        double Cv,
        double sigma,
        double beta,
        double Cr,
        double theta,
        double alpha_g
):
    # TODO: step for RL environment
    cdef:
        int i,j,k
        npc.ndarray[npc.double_t, ndim = 1] reward = np.zeros(n,dtype=np.float64)#[0.0 for i in range(n)]
        bint done = False  #[False for i in range(8)]
        double rwd, v_max = 18
        double[:] action_l = np.empty(2,dtype=np.float64)
        double[:,::1] coll_reward = np.empty((n+1,n+1),dtype=np.float64)
    actions = nparray_to_double_array(actions_)
    #Update state
    action_l[0] = 2 * (generate_randoms() - 0.5)
    action_l[1] = 2 * (generate_randoms() - 0.5)
    state[0,3] += action_l[1] * rollmax
    state[0,3] = clip(state[0,3], -M_PI / 12, M_PI / 12)
    state[0,4] += action_l[0] * vmax
    state[0,4] = np.clip(state[0,4], 12, 18)
    # #update v & roll
    for j in range(1, n+1):
        state[j,3] += actions[j - 1, 1] * rollmax
        state[j,4] += actions[j - 1, 0] * vmax
        state[j,3] = clip(state[j,3], -M_PI / 12, M_PI / 12)
        state[j,4] = clip(state[j,4], 12, 18)
    for k in range(n+1):
        state[k,2] += -(alpha_g / (state[k,4] + 0.0001)) * tan(state[k,3])
        while cabs(state[k,2]) > 2 * np.pi:
            if state[k,2] > 0:
                state[k,2] -= 2 * M_PI
            if state[k,2] < 0:
                state[k,2] += 2 * M_PI
        state[k,0] += state[k,4] * cos(state[k,2]) * 0.5
        state[k,1] += state[k,4] * sin(state[k,2]) * 0.5
    collision_list = crash_jud(state, n, wing_length, air_length)
    if np.sum(collision_list)>0:
        done = True
    for i in range(n):
        rwd = L_G_CSreward(state, i+1, n, Cv, sigma, beta, Cr, theta)*1000
        rwd /= get_r_sup(Cv, Cr, sigma, beta, v_max)
        reward[i] += rwd - collision_list[i]
    cdef list info = []
    for i in range(n+1):
        info.append(state[i])
    return state, reward, done, info

