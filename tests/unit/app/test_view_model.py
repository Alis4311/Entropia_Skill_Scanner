from entropia_skillscanner.core import SkillRow
from entropia_skillscanner.view_model import SkillScannerViewModel


def test_status_subscribers_receive_updates_without_duplicates():
    vm = SkillScannerViewModel()
    seen = []

    vm.subscribe("status", lambda v: seen.append(v))

    vm.set_status("running")
    vm.set_status("running")  # should not notify again
    vm.set_status("done")

    assert seen == ["", "running", "done"]


def test_rows_subscription_receives_initial_and_appended_rows():
    vm = SkillScannerViewModel()
    seen = []

    vm.subscribe("rows", lambda rows: seen.append(list(rows)))

    first = SkillRow(name="Combat", value=10.0, added="t0")
    second = SkillRow(name="Engineering", value=20.0, added="t1")

    vm.append_rows([first])
    vm.append_rows([])  # ignored
    vm.append_rows([second])

    assert seen == [
        [],
        [first],
        [first, second],
    ]


def test_warnings_subscription():
    vm = SkillScannerViewModel()
    seen = []

    vm.subscribe("warnings", lambda warnings: seen.append(tuple(warnings)))

    vm.set_warnings(["bad data"])
    vm.set_warnings(["bad data"])  # unchanged should not notify
    vm.set_warnings(["bad data", "missing skill"])

    assert seen == [
        (),
        ("bad data",),
        ("bad data", "missing skill"),
    ]
