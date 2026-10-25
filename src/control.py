"""Discrete PI controller with output saturation and back-calculation anti-windup."""


class PI:
    def __init__(self, kp, ki, ts, lo=-1e9, hi=1e9, kaw=None, x0=0.0):
        self.kp, self.ki, self.ts = kp, ki, ts
        self.lo, self.hi = lo, hi
        self.kaw = (ki / kp if kp else 1.0) if kaw is None else kaw
        self.x = x0          # integrator state (output units)

    def step(self, e, ff=0.0, lo=None, hi=None):
        lo = self.lo if lo is None else lo
        hi = self.hi if hi is None else hi
        u_raw = self.kp * e + self.x + ff
        u = hi if u_raw > hi else lo if u_raw < lo else u_raw
        self.x += self.ts * (self.ki * e + self.kaw * (u - u_raw))
        return u

    def reset(self, x0=0.0):
        self.x = x0
