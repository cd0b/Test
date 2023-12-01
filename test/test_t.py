from t import t

def test_t():
    input_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    result_list = t(input_list)
    assert(result_list == [2.0,4.0,6.0,8.0,10.0])