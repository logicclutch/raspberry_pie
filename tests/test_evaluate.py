from anpr.evaluate import Score, read_labels


def test_score_precision_recall():
    s = Score()
    s.add("a.jpg", {"MH12AB1234"}, ["MH12AB1234"])
    s.add("b.jpg", {"KA01MJ0001"}, ["KA01MJ0007"])  # wrong plate shown
    s.add("c.jpg", {"TN09A1234"}, [])  # missed: hurts recall, not precision
    s.add("d.jpg", set(), ["DL3CAB1234"])  # plate invented on an empty image
    assert s.shown == 3 and s.correct == 1 and s.expected == 3
    assert s.precision == 1 / 3
    assert s.recall == 1 / 3
    assert len(s.errors) == 2
    assert "FAIL" in s.report(0.995)


def test_duplicates_count_once():
    s = Score()
    s.add("v.mp4", {"MH12AB1234"}, ["MH12AB1234", "MH12AB1234"])
    assert s.correct == 1 and s.shown == 2


def test_empty_run_passes_precision():
    assert Score().precision == 1.0


def test_read_labels(tmp_path):
    (tmp_path / "labels.csv").write_text(
        "image,plate\nimg/a.jpg,mh 12 ab 1234\n# comment\nimg/b.jpg,\nv.mp4,MH12AB1234;KA01MJ0001\n"
    )
    rows = read_labels(tmp_path / "labels.csv")
    assert rows[0] == ((tmp_path / "img/a.jpg").resolve(), {"MH12AB1234"})
    assert rows[1][1] == set()
    assert rows[2][1] == {"MH12AB1234", "KA01MJ0001"}
