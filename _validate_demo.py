from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app/demo_app.py", default_timeout=300)
at.run()
print("initial exception?", bool(at.exception))
for e in at.exception: print("EXC:", repr(e)[:400])
# live metrics
labels = [m.label for m in at.metric]
print("metrics present:", labels[:8])
# click calibrate in backtest tab
cb = [b for b in at.button if b.label and "sigma calibration" in b.label]
print("calibrate button:", bool(cb))
if cb:
    cb[0].click().run()
    print("after calibrate exception?", bool(at.exception))
    for e in at.exception: print("EXC2:", repr(e)[:400])
print("DONE")
