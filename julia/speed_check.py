#!/home/sdeshmukh/anaconda3/bin/python
# Quickly check speed of 3D JCRN based on test run avg time
import sys

def main():
  for arg in sys.argv[1:]:
    test_run_time = float(arg)  # presumed in seconds
    test_size = 4200  # number of points in test
    snapshot_size = 150 * 140 * 140  # standard 3D dwarf model
    total_time_s = test_run_time / test_size * snapshot_size
    total_time_m = total_time_s / 60
    total_time_h = total_time_m / 60
    print(f"Original time: {test_run_time:.1f} [s]\n\tEstimated time: {int(total_time_s)} [s] = {int(total_time_m)} [m]  = {total_time_h:.2f} [h]")

if __name__ == "__main__":
  main()
