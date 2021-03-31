import sys
sys.path.insert(1, '/Users/duongp/ganspace-online')
from export import GanModel

car = GanModel(model='StyleGAN2', class_name='car', layer='style', n=1_000_000, b=10_000)