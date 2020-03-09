files = ['img12.png', 'img10.png', 'img02.png', 'img1.png', 'IMG01.GIF', 'img2.JPG']

import copy
def soltion(files):
    _answer = []
    for file in files:
        file1 = copy.deepcopy(file)
        file2 = file.lower()
        _factor = 0
        first, second, other = '', '', ''
        for word in file2:
            if _factor == 0 and ('0' > word or word > '9'):
                first += word

            elif '0' <= word <= '9':
                if _factor == 0:
                    _factor = 1
                    second += word

                elif _factor == 1:
                    second += word

                else:
                    other += word
            elif _factor == 1 and ('0' > word or word > '9'):
                _factor = 2
                other += word
            else:
                other += word

        _answer.append([file1, first, second, other])

    _answer.sort(key=lambda x: [x[1], int(x[2])])
    answer = [x[0] for x in _answer]

    return answer
