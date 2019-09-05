from urllib import request

from assets.models.download_dlib_shape_predictor_68 import main


def test_check_dlib_landmarks_url(check_url):
    resp = request.urlopen(check_url["dlib"])

    assert 200 == resp.status
    
    print("Start to download dlib lanmarks models")
    main()
    print("Done")
