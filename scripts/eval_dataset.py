"""Golden evaluation dataset for the Digimon RAG pipeline.

Level naming follows DAPI conventions (not Western fandom names):
  Baby I   = Fresh,       Baby II = In-Training
  Child    = Rookie,      Adult   = Champion
  Perfect  = Ultimate,    Ultimate = Mega

All Digimon here are confirmed present in the default 100-Digimon ingest.
"""

EVAL_DATASET = [
    {
        "question": "What level is Agumon and what type is it?",
        "ground_truth": "Agumon is a Child level Digimon of the Reptile type with Vaccine attribute.",
        "expected_digimon": "Agumon",
    },
    {
        "question": "What are Gabumon's main characteristics?",
        "ground_truth": "Gabumon is a Child level Digimon of the Reptile type with Data attribute.",
        "expected_digimon": "Gabumon",
    },
    {
        "question": "Describe the Koromon Digimon.",
        "ground_truth": "Koromon is a Baby II level Digimon of the Lesser type with Data attribute.",
        "expected_digimon": "Koromon",
    },
    {
        "question": "What kind of Digimon is Greymon?",
        "ground_truth": "Greymon is an Adult level Digimon of the Dinosaur type with Vaccine attribute.",
        "expected_digimon": "Greymon",
    },
    {
        "question": "What type and attribute is Metal Etemon?",
        "ground_truth": "Metal Etemon is an Ultimate level Digimon of the Cyborg type with Virus attribute.",
        "expected_digimon": "Metal Etemon",
    },
    {
        "question": "What level is MetalGreymon?",
        "ground_truth": "MetalGreymon is a Perfect level Digimon of the Cyborg type.",
        "expected_digimon": "MetalGreymon",
    },
    {
        "question": "Describe the characteristics of Angemon.",
        "ground_truth": "Angemon is an Adult level Digimon of the Angel type with Vaccine attribute.",
        "expected_digimon": "Angemon",
    },
    {
        "question": "What is Patamon?",
        "ground_truth": "Patamon is a Child level Digimon of the Mammal type with Data attribute.",
        "expected_digimon": "Patamon",
    },
    {
        "question": "What type is Betamon?",
        "ground_truth": "Betamon is a Child level Digimon of the Amphibian type with Virus attribute.",
        "expected_digimon": "Betamon",
    },
    {
        "question": "What are the characteristics of Botamon?",
        "ground_truth": "Botamon is a Baby I level Digimon of the Slime type with Free attribute.",
        "expected_digimon": "Botamon",
    },
]
