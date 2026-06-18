"""Console entry point for the IaR demo (the ``run`` script, per the Volue standard).

The primary interface is the Streamlit dashboard. This ``main`` exists so the package
exposes a console-script entry point; it prints how to launch the dashboard and rebuild
the synthetic database.
"""


def main() -> int:
    print("Imbalance at Risk - synthetic demo")
    print("Launch the dashboard:        streamlit run app/dashboard.py")
    print("Rebuild synthetic database:  python scripts/seed_synthetic_demo.py --area NO2 --days 30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
