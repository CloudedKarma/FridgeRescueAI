"""Conservative normalization for structured Food.com ingredient names."""
import re
import unicodedata

GROUPS = {
    'garlic': ['garlic clove','garlic cloves','fresh garlic','fresh garlic clove','fresh garlic cloves','minced garlic clove','minced garlic cloves','clove garlic','cloves garlic'],
    'green onion': ['scallion','scallions','spring onion','spring onions','green onions'],
    'chickpea': ['chickpeas','garbanzo bean','garbanzo beans'],
    'water': ['cold water','warm water','hot water','boiling water','tap water'],
    'black pepper': ['ground black pepper','fresh ground black pepper','freshly ground black pepper'],
    'lemon juice': ['lemon, juice of','lemons, juice of','fresh lemon juice'],
    'lime juice': ['lime, juice of','limes, juice of','fresh lime juice'],
    'powdered sugar': ["confectioners' sugar",'icing sugar'],
    'ketchup': ['catsup','tomato ketchup'],
    'yogurt': ['yoghurt'],
    'zucchini': ['courgette','courgettes'],
    'eggplant': ['aubergine','aubergines'],
    'soy sauce': ['soya sauce'],
    'baking soda': ['bicarbonate of soda'],
    'self-rising flour': ['self raising flour','self-raising flour','self rising flour'],
}
ALIASES = {a:b for b, aa in GROUPS.items() for a in aa}
# Only known food nouns; do not strip arbitrary trailing s or modifiers.
PLURALS = dict(zip('eggs onions carrots potatoes tomatoes bananas apples breasts thighs drumsticks wings fillets chops crusts shells beans peas cloves mushrooms peppers leaves berries strawberries blueberries raspberries blackberries cherries peaches pears oranges lemons limes avocados mangoes nuts almonds walnuts pecans cashews pistachios tortillas crackers steaks cucumbers eggplants seeds yolks whites sticks florets ribs stalks noodles'.split(), 'egg onion carrot potato tomato banana apple breast thigh drumstick wing fillet chop crust shell bean pea clove mushroom pepper leaf berry strawberry blueberry raspberry blackberry cherry peach pear orange lemon lime avocado mango nut almond walnut pecan cashew pistachio tortilla cracker steak cucumber eggplant seed yolk white stick floret rib stalk noodles'.split()))
# These exact nonfood entries are excluded from the shopping/food model, not the original recipe.
EQUIPMENT = {'aluminum foil','parchment paper','wooden skewers','bamboo skewers','bamboo skewer','skewers','skewer','toothpick','toothpicks','wooden popsicle sticks','cheesecloth','reynolds wrap foil'}
COMPOSITES = {'salt and pepper': ('salt','pepper'), 'salt & pepper': ('salt','pepper'), 'salt and black pepper': ('salt','black pepper'), 'salt & freshly ground black pepper': ('salt','black pepper'), 'salt and freshly ground black pepper': ('salt','black pepper')}

def normalize_name(value):
    s=' '.join(unicodedata.normalize('NFKC',str(value)).lower().strip().split())
    s=s.replace('’', "'")
    s=re.sub(r'^(\d+)\s*[- ]\s*inch\s+',r'\1-inch ',s)
    if s in ALIASES: return ALIASES[s]
    words=s.split()
    if words and words[-1] in PLURALS: words[-1]=PLURALS[words[-1]]
    s=' '.join(words)
    return ALIASES.get(s,s)

def normalize_items(items):
    out=set()
    for item in items:
        s=' '.join(str(item).lower().strip().split())
        if not s or s in EQUIPMENT: continue
        for part in COMPOSITES.get(s,(s,)):
            n=normalize_name(part)
            if n: out.add(n)
    return out
