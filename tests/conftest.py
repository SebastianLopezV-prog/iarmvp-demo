from pytest_socket import disable_socket


# By default, tests have no network access. This keeps the suite hermetic and ensures we
# are testing our own code, not external connections. The demo is synthetic and offline
# anyway. To allow sockets for a specific test, decorate it with
# ``@pytest.mark.enable_socket``.
def pytest_runtest_setup():
    disable_socket()
