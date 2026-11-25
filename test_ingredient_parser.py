import unittest
from Archived.ingredient_parser import parse_line,normalize
class ParserTests(unittest.TestCase):
    def test_recovery(self):
        for raw,expected in [('1 (17 oz.) pkg. spaghetti, cooked as directed','spaghetti'),('2 T Chia Seeds','chia seed'),('1/4 Cup Erythitrol, powdered','erythritol'),('3 cloves garlic','garlic'),('2 large eggs','egg'),('1/2 cup cold water','water'),('1 cup garlic powder','garlic powder')]:
            with self.subTest(raw=raw): self.assertEqual(parse_line(raw),(expected,'parsed'))
    def test_alternatives(self):
        self.assertEqual(parse_line('3 ounces guanciale, or pancetta')[1],'review')
    def test_header(self):
        self.assertEqual(parse_line('For the crust'),(None,'heading'))
    def test_synonyms(self):
        self.assertEqual(normalize('scallions'),'green onion')
        self.assertNotEqual(normalize('garlic powder'),normalize('garlic'))
if __name__=='__main__': unittest.main()
