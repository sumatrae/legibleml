#encoding=utf-8
import unittest
from email_classification import *

class MyTestCase(unittest.TestCase):

    def test_split2words(self):
        content = "我是中国人,人人有责。  中国好人有好报"
        words = split2words(content)
        print(words)

if __name__ == '__main__':
    unittest.main()
