def pytest_addoption(parser):
    parser.addoption("--work_dir", action="store", type=str)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    work_dir_value = metafunc.config.option.work_dir
    if "work_dir" in metafunc.fixturenames:
        metafunc.parametrize("work_dir", [work_dir_value])
