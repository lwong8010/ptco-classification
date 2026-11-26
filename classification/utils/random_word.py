import random

def get_random_word_combo():
    # List of animal names
    animals = [
        "cat", "dog", "bird", "fish", "horse", "cow", "sheep", "goat", "pig", 
        "elephant", "lion", "tiger", "bear", "monkey", "zebra", "giraffe", "hippo", 
        "penguin", "koala", "fox", "wolf", "rabbit", "deer", "moose", "elk", "rhino", 
        "buffalo", "shark", "whale", "dolphin", "sealion", "monkey", "gorilla", "chimpanzee",
        "aardvark", "anteater", "bat", "beetle", "bumblebee", "camel", "caterpillar", 
        "cheetah", "chinchilla", "clam", "cobra", "cockroach", "crab", "crocodile", 
        "deer", "dinosaur", "dolphin", "duck", "eagle", "elephant", "emu", "falcon", 
        "ferret", "finch", "fish", "flamingo", "fly", "fox", "frog", "goat", "goldfish", 
        "gorilla", "grasshopper", "hawk", "hippopotamus", "hornet", "horse", "hummingbird", 
        "ibex", "jackal", "jaguar", "jay", "kangaroo", "kiwi", "koala", "ladybug", 
        "lamprey", "leopard", "lion", "lizard", "lobster", "locust", "louse", "lynx", 
        "mandrill", "marmot", "meerkat", "monkey", "moose", "mosquito", "mouse", "mule", 
        "narwhal", "newt", "nightingale", "octopus", "opossum", "ostrich", "owl", "oyster", 
        "panda", "panther", "parrot", "peacock", "pelican", "penguin", "pheasant", "pig", 
        "piranha", "platypus", "porcupine", "possum", "prawn", "primate", "ptarmigan", "quail", 
        "rabbit", "racoon", "ram", "rat", "raven", "reindeer", "rhinoceros", "salmon", 
        "scorpion", "sealion", "seal", "shark", "sheep", "snail", "snake", "spider", 
        "squid", "squirrel", "starfish", "stingray", "stork", "swan", "tiger", "toad", 
        "tortoise", "turkey", "turtle", "urchin", "viper", "walrus", "wasp", "weasel", 
        "whale", "wildcat", "wildebeest", "wolf", "worm", "wren", "yak", "zebra"
    ]

    # List of adjectives
    adjectives = [
        "ambitious", "apathetic", "boisterous", "callous", "charismatic", "diligent",
        "eccentric", "fastidious", "gregarious", "haughty", "impetuous", "jubilant",
        "luminous", "meticulous", "nefarious", "obstinate", "peculiar", "quixotic",
        "resilient", "sagacious", "tenacious", "ubiquitous", "vivacious", "whimsical",
        "zealous", "adamant", "benevolent", "capricious", "delirious", "ethereal",
        "ferocious", "grandiose", "harmonious", "illustrious", "judicious", "kinetic",
        "lavish", "malevolent", "notorious", "opulent", "pristine", "querulous",
        "rambunctious", "serendipitous", "turbulent", "unwavering", "venerable", "wistful",
        "xenial", "yearning", "ardent", "brilliant", "cryptic", "dexterous", "eloquent",
        "flamboyant", "gallant", "heinous", "imperial", "jovial", "keen", "lucid",
        "mystical", "nimble", "obscure", "pensive", "quintessential", "radiant", "stoic",
        "tactful", "urbane", "valiant", "wanderlust", "xenodochial", "yielding", "zesty",
        "astute", "baroque", "celestial", "defiant", "effervescent", "furtive", "garish",
        "halcyon", "idyllic", "jocular", "kaleidoscopic", "laconic", "mercurial", "nebulous",
        "ornate", "pastoral", "quaint", "ravenous", "serene", "tranquil", "utopian",
        "verdant", "winsome", "youthful", "zealous", "aesthetic", "blithe", "cosmic",
        "dynamic", "ephemeral", "fanciful"
    ]
    
    # Randomly select an animal and an adjective
    random_animal = random.choice(animals)
    random_adjective = random.choice(adjectives)
    
    # Combine the adjective and animal into a word combo
    random_word_combo = f"{random_adjective}_{random_animal}"
    
    return random_word_combo