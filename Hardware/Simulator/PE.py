import numpy as np

class PE_array:
    def __init__(self):
        # 64 x 64 PE array
        self.res = [[0 for i in range(64)] for j in range(64)]

    def reg_Q(self, temp_Q):
        self.Q = temp_Q

    def reg_K(self, temp_K):
        self.K = temp_K

    def reg_V(self, temp_V):
        self.V = temp_V

    def reg_attn(self, temp_attn):
        self.attn = temp_attn

    def reg_index(self, temp_index):
        self.index = temp_index

    def cal_attn_map(self):
        assert self.Q is not None
        assert self.K is not None
        assert len(self.Q) == len(self.K)

        # print('Shape of Q: ', self.Q.shape)
        # print('Shape of K: ', self.K.shape)

        cycle = 0
        for k in range(len(self.Q)):
            self.res[0][0] += self.Q[k] * self.K[k]
            cycle += 1

        return self.res[0][0], cycle

    def cal_V_update(self):
        assert self.attn is not None
        assert self.V is not None
        # print('Shape of attn_map: ', self.attn.shape)
        # print('Shape of V: ', self.V.shape)

        assert len(self.attn[0]) == len(self.V[0])

        cycle = 0
        for i in range(len(self.attn)):
            for j in range(len(self.V)):
                for k in range(len(self.V[0])):
                    self.res[i][j] += self.attn[i][k] * self.V[j][k]
                    cycle += 1

        return np.array(self.res), cycle

    def reset_res(self):
        self.res = [[0 for i in range(64)] for j in range(64)]

    def store_res(self):
        cycle = 1
        return cycle

    def store_res_V(self):
        cycle = len(self.attn) * len(self.V) // 64 # assume 64 number paralleism for one cycle
        return cycle

