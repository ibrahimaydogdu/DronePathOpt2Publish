"""Wrapper: run run_bridge_real.py at n=12 at below-normal priority (user asked to leave CPU free)."""
import ctypes, runpy, sys, os
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
os.environ.setdefault("NREQ", "12"); os.environ.setdefault("NW", "2"); os.environ.setdefault("OUT", "bridge_real_n12.csv")
sys.argv = ["run_bridge_real.py", "10", "20000"]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_bridge_real.py"), run_name="__main__")
