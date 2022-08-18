class SingleArmDetector:

  def __init__(self, M, eps, h):

    # number of samples needed to initialize the reference value
    self.M = M

    # reference value
    self.reference = 0

    # epsilon value, parameter of change detection formula 
    self.eps = eps

    # threshold to detect the change
    self.h = h

    # g values that will be computed by the algorithm
    self.g_plus = 0
    self.g_minus = 0

    # number of rounds executed
    self.t = 0

  def update(self, sample):
    


    self.t += 1

    if self.t <= self.M:
      self.reference += sample/self.M
      return False
    else:
      self.reference = (self.reference*(self.t-1) + sample)/self.t
      s_plus = (sample - self.reference) - self.eps
      s_minus = -(sample - self.reference) - self.eps

      self.g_plus = max(0, self.g_plus + s_plus)
      self.g_minus = max(0, self.g_minus + s_minus)

      if self.g_minus > self.h or self.g_plus > self.h:
        print("REFERENCE: ", self.reference, "Sample:", sample)
        self.reset()
        return True
      return False

  def reset(self):
    self.t = 0
    self.g_minus = 0
    self.g_plus = 0
    self.reference = 0
