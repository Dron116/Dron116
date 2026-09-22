from app.plates import candidates_from_ocr, format_plate, is_valid_plate, normalize_plate, to_cyrillic


def test_latin_lookalikes_become_cyrillic():
    assert to_cyrillic("A123BC777") == "А123ВС777"
    assert normalize_plate("A123BC777") == "А123ВС777"


def test_spacing_and_noise():
    assert normalize_plate("А 123 ВС 777") == "А123ВС777"
    assert normalize_plate("номер: А123ВС777 конец") == "А123ВС777"


def test_region_two_and_three_digits():
    assert normalize_plate("К456ЕМ99") == "К456ЕМ99"
    assert normalize_plate("К456ЕМ199") == "К456ЕМ199"


def test_digit_letter_coercion_by_position():
    assert normalize_plate("O001OO77") == "О001ОО77"
    assert normalize_plate("A12EBC777") is None


def test_format_plate():
    assert format_plate("А123ВС777") == "А 123 ВС 777"
    assert format_plate("М007ОО77") == "М 007 ОО 77"


def test_invalid_rejected():
    assert normalize_plate("HELLO") is None
    assert normalize_plate("12345678") is None
    assert not is_valid_plate("XYZ")


def test_candidates_pick_embedded_plate():
    found = candidates_from_ocr("CAM1 A123BC777 OK")
    assert found == ["А123ВС777"]
