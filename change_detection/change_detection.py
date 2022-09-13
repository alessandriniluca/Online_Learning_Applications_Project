from change_detection.single_arm_detector import SingleArmDetector

class ChangeDetection:
  def __init__(self, n_arms, M, eps, h):
    self.n_arms = n_arms
    self.detectors = [SingleArmDetector(M, eps, h) for arm in range(n_arms)]

  def update(self, arm, sample):
    return self.detectors[arm].update(sample)
