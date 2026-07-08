"""
morphological_catalog.py — Enhanced Morphological Catalog for RAVL classification.
Contains detailed freshwater AND marine plankton descriptions.
Written with input from an AI language model, reviewed against taxonomic references.
"""

# Freshwater species (ZooLake / Chen OOD)
FRESHWATER_CATALOG = {
    "aphanizomenon": "Filamentous cyanobacterium forming elongated straight or slightly curved trichomes. Cells cylindrical (3-5 μm wide), may form bundles or rafts. Blue-green color. No heterocysts visible at this magnification. Often confused with Planktothrix but Aphanizomenon forms flat ribbon-like bundles.",
    "asplanchna": "Large predatory rotifer (500-2000 μm) with transparent sac-like body. No visible lorica (shell). Prominent corona (crown of cilia) at anterior end. Internal organs (mastax, gut) clearly visible through transparent body wall. Body shape changes constantly due to active swimming.",
    "asterionella": "Star-shaped colonial diatom. Cells lanceolate (lance-shaped, 50-100 μm long), attached at one end forming radiating star colonies of 4-8 cells. Siliceous cell wall with fine striations. One of the most recognizable freshwater diatoms.",
    "bosmina": "Small cladoceran (300-500 μm) with characteristic long antennae and a downward-curving rostrum (beak-like projection). Oval body enclosed in bivalve shell (carapace). Second antennae used for jumping. Brood chamber visible in females. Distinguished from Daphnia by smaller size and curved rostrum.",
    "brachionus": "Planktonic rotifer with prominent lorica (shell). Lorica typically oval with 2-4 anterior spines. Corona with two large wheel organs used for feeding. Body length 150-300 μm. Posterior foot may be present. Common in eutrophic waters.",
    "ceratium": "Dinoflagellate with 2-4 horn-like projections. One forward horn (hypotheca), 1-3 backward horns (epitheca). Visible cingulum (groove) around cell middle. Chloroplasts present (golden-brown). Cell body 40-80 μm. Horns can be straight or curved depending on species and conditions.",
    "chaoborus": "Phantom midge larva (Diptera). Transparent cylindrical body (5-10 mm) with two pairs of air sacs (hydrostatic organs) visible as dark oval spots. Dark eye spots prominent. No legs in larval stage. Body segmentation visible. Moves by rapid jerking motions.",
    "collotheca": "Sessile rotifer with trumpet-shaped or vase-shaped body. Corona modified into a funnel for capturing food particles. Transparent body wall. Body length 200-500 μm. Often attached to substrates or other organisms.",
    "conochilus": "Colonial rotifer forming spherical colonies (1-3 mm diameter) of individuals attached to a common gelatinous base. Each individual has prominent corona with wheel organs. Colony swims as a unit. Distinguished from other colonial rotifers by spherical shape.",
    "copepod_skins": "Transparent exuviae (molted exoskeletons) of copepods. Elongated body shape with visible segmentation. Fragile, often fragmented. Antennae and swimming legs may be partially preserved. No internal organs visible (empty shell).",
    "cyclops": "Cyclopoid copepod with elongated body (1-2 mm), long first antennae (reaching past midpoint of body), and prominent egg sacs. Single median eye. Body divided into prosome (anterior) and urosome (posterior). Fifth pair of legs reduced. Common in freshwater.",
    "daphnia": "Large cladoceran (water flea, 1-5 mm) with prominent bivalve shell (carapace). Large second antennae for jumping/ swimming. Visible brood chamber in females (contains developing embryos). Single compound eye. Transparent body allows internal organs to be seen. One of the most common freshwater zooplankton.",
    "daphnia_skins": "Transparent exuviae (molted exoskeletons) of Daphnia. Bivalve shell shape clearly visible. Fragile, often with visible brood chamber outline. Antennae may be partially preserved. Empty interior.",
    "diaphanosoma": "Transparent cladoceran with elongated body (1-2 mm). Thin shell allows internal organs to be clearly visible. Long second antennae. No head shield (distinguishes from Daphnia). Body more elongated than Daphnia.",
    "diatom_chain": "Chain of diatom cells linked together. Cells disc-shaped or cylindrical, connected by organic threads or silica spines. Chain length variable (2-20 cells). Cell wall (frustule) shows fine punctae or striations.",
    "dinobryon": "Colonial golden-brown alga forming tree-like or vase-shaped colonies. Individual cells enclosed in loricae (vases) that branch dichotomously. Characteristic branching pattern. Cells have two flagella. Colony can be several mm long.",
    "dirt": "Non-biological debris. Irregular shape, opaque, variable color (brown, black, grey). May include mineral particles, plant fragments, or organic detritus. No biological structure visible. Variable size and shape.",
    "eudiaptomus": "Calanoid copepod with elongated body (1-2 mm) and long first antennae (reaching past body). Characteristic asymmetric genital segment (males). Red pigmentation possible (hemoglobin). Egg sacs in females. Fifth pair of legs biramous.",
    "filament": "Thin elongated structure, may be algal filament, plant fiber, or cyanobacterial trichome. Variable length (50-500 μm), often curved or tangled. No clear cellular structure visible. Could be fragments of larger organisms.",
    "fish": "Fish larva or egg. Larvae: elongated body (2-10 mm) with visible yolk sac, eyes prominent, body transparent. Eggs: spherical (0.5-2 mm) with embryo visible inside. Melanophores (pigment cells) may be present on larvae.",
    "fragilaria": "Chain-forming diatom. Cells tabular (rectangular in girdle view, 20-50 μm), linked into flat ribbons. Siliceous cell wall with fine striations. Chains can be several hundred μm long. One of the most common freshwater diatoms.",
    "hydra": "Freshwater cnidarian with tubular body (1-10 mm) and tentacles (4-12 tentacles arranged around mouth). Typically attached at base. Green color from symbiotic algae (Chlorella) possible. Tentacles can be extended or contracted. Body wall transparent.",
    "kellicottia": "Spined rotifer with elongated body and prominent posterior spines (2-4 spines). Corona with two wheel organs. Transparent lorica. Body length 200-400 μm. Spines used for defense and flotation.",
    "keratella_cochlearis": "Small planktonic rotifer (100-200 μm) with asymmetric lorica. Posterior spine variable in length (can be very long). Corona with two wheel organs. One of the most common freshwater rotifers. Distinguished from K. quadrata by asymmetric lorica.",
    "keratella_quadrata": "Planktonic rotifer with quadrangular lorica. Four anterior spines and one posterior spine. Corona with two wheel organs. Body length 150-300 μm. More robust than K. cochlearis.",
    "leptodora": "Large predatory cladoceran (>10 mm, one of the largest freshwater zooplankton). Transparent body with prominent raptorial first legs (for catching prey). Large compound eye. Jellyfish-like appearance. Body almost completely transparent.",
    "maybe_cyano": "Possibly cyanobacterial colony or trichome. Variable morphology, may be filamentous or colonial. Green or blue-green color. Could be Microcystis, Anabaena, or other cyanobacterial genera. Definitive identification requires higher magnification.",
    "nauplius": "Early larval stage of crustaceans (copepods, cladocerans). Small (100-300 μm), pear-shaped with three pairs of appendages (antennules, antennae, mandibles). Single median eye. Body not yet segmented. Active swimmer.",
    "paradileptus": "Ciliate protozoan with elongated body and prominent cilia. Fast-swimming. Characteristic body shape with anterior narrowed region. Body length 100-300 μm. Cilia arranged in rows.",
    "polyarthra": "Planktonic rotifer with paddle-like appendages (epipodia) for swimming. Small (100-250 μm), transparent. Corona with two wheel organs. Epipodia give distinctive appearance. Common in lakes and ponds.",
    "rotifers": "General rotifer category. Microscopic animals (100-500 μm) with corona (crown of cilia) used for feeding and locomotion. Variable body forms (asymmetrical, spherical, elongated). Pseudocoelomate. Distinguished from other zooplankton by corona.",
    "synchaeta": "Planktonic rotifer with elongated body and prominent lateral antennae. No lorica (shell-less). Corona with two wheel organs. Body length 200-500 μm. Active swimmer. Distinguished from other loricate rotifers by absence of shell.",
    "trichocerca": "Planktonic rotifer with asymmetric body (one side curved, other straight). Corona with two wheel organs. Small (100-200 μm). Asymmetric lorica gives distinctive appearance. Common in freshwater.",
    "unknown": "Organism that cannot be identified to class level. Variable morphology. May be fragment, artifact, or unfamiliar organism. No definitive diagnostic features visible at this magnification.",
    "unknown_plankton": "Planktonic organism of uncertain taxonomic affiliation. Microscopic, aquatic morphology but unknown class. Could be fragment of larger organism or unusual morphological variant.",
    "uroglena": "Colonial golden-brown alga forming spherical colonies (0.5-3 mm). Individual cells with two flagella embedded in gelatinous matrix. Fishy odor when abundant. Colonial matrix transparent. Cells arranged radially.",
}

# Marine species (IFCB / ZooScan)
MARINE_CATALOG = {
    "amphipoda": "Small crustacean (1-10 mm) with laterally compressed body. Distinct head, thorax, and abdomen. Seven pairs of walking legs (pereopods). Body segmented. Antennae prominent. Distinguished from other crustaceans by laterally compressed body and jumping behavior.",
    "annelida": "Segmented worm (polychaete larvae). Elongated body with visible segmentation. Chaetae (bristles) may be visible on parapodia. Body length variable (0.5-5 mm). Head region may have tentacles or palps.",
    "appendicularia": "Tunicate (larvacean) with oikopleuran house. Transparent body (1-5 mm) with tail (notochord). House is gelatinous, secreted by the animal, used for filter feeding. House often more visible than the animal itself.",
    "calanoida": "Calanoid copepod with elongated body (1-5 mm). Long first antennae (reaching past body). Body divided into prosome and urosome. Egg sacs in females. Fifth pair of legs biramous. Most common copepod group in marine plankton.",
    "ceratium": "Marine dinoflagellate with horn-like projections. Similar to freshwater Ceratium but marine species may have different horn configurations. Cingulum visible. Golden-brown chloroplasts. Cell body 40-100 μm.",
    "chaetoceros": "Chain-forming marine diatom. Cells cylindrical or barrel-shaped (10-50 μm), linked into chains. Long setae (spines) extending from each cell corner. Setae can be several times cell length. One of the most abundant marine diatoms.",
    "chaetognatha": "Arrow worm (2-10 mm). Elongated, transparent body with lateral fins. Grasping spines at anterior end. Horizontal fins along body. Eyespot visible. Predatory. Body stiff due to turgor pressure.",
    "coscinodiscus": "Centric diatom with disc-shaped cell (50-200 μm diameter). Radial symmetry. Siliceous cell wall with ornate pattern of areolae (pores). Can be very large for a diatom. Girdle view shows box-like shape.",
    "doliolida": "Transparent barrel-shaped tunicate (1-10 mm). Muscle bands visible around body. Siphons at anterior and posterior ends. Life cycle includes alternation of generations. Body wall transparent, internal structures visible.",
    "euplotes": "Hypotrich ciliate with dorsoventrally flattened body. Ventral surface has compound ciliary structures (cirri) used for walking and swimming. Body length 50-200 μm. Distinctive shape with flattened ventral surface.",
    "guinardia": "Marine diatom forming chains. Cells cylindrical (20-100 μm diameter), linked end-to-end. Cell wall with fine striations. Chains can be very long (mm to cm). Common in temperate marine waters.",
    "mesodinium": "Ciliate protozoan with symbiotic algae. Body spherical to ovoid (30-80 μm). Prominent cilia at anterior end. Contains red-pigmented cryptophyte endosymbionts. Fast swimmer. Can form red tides.",
    "noctiluca": "Large dinoflagellate (200-2000 μm). Spherical to oval body with prominent tentacle. Bioluminescent. Single large nucleus. No typical dinoflagellate plates. One of the largest single-celled plankton.",
    "oithonidae": "Small cyclopoid copepod (0.5-2 mm). Short first antennae. Egg sacs carried by females. Fast-swimming. Body more compact than calanoid copepods. Common in coastal marine waters.",
    "ostracoda": "Seed shrimp with bivalve carapace (0.5-2 mm). Body enclosed in two shell valves. Appendages protrude from shell opening. No visible segmentation externally. Carapace may be transparent or opaque.",
    "protoperidinium": "Dinoflagellate with thecal plates. Body spherical to ovoid (30-100 μm). Two flagella. Plate pattern diagnostic for genus. No horns (distinguishes from Ceratium). May have apical horn or antapical spines.",
    "pseudo-nitzschia": "Pennate diatom forming chains. Cells needle-shaped (50-200 μm long), linked in overlapping chains. Raphe (slit) visible. Can produce domoic acid (neurotoxin). One of the most common marine diatoms.",
    "rhizosolenia": "Marine diatom with elongated cylindrical cells (50-500 μm). Cells linked in chains. Long setae at cell ends. Cell wall with fine striations. Some species have very long cells (mm scale).",
    "salpida": "Transparent tunicate (1-100 mm). Barrel-shaped body with muscle bands. Filter feeder. Life cycle includes aggregate chains (colonies) and solitary individuals. Body wall completely transparent.",
    "strombidium": "Oligotrich ciliate with prominent oral cilia. Body spherical to ovoid (50-200 μm). Girdle of somatic cilia around body equator. Fast swimmer. Common marine ciliate.",
    "thalassiosira": "Centric diatom forming chains. Cells disc-shaped (10-100 μm diameter), linked by organic threads. Radial symmetry in valve view. One of the most abundant marine diatoms. Fine punctae on cell wall.",
    "tintinnopsis": "Tintinnid ciliate with lorica (shell). Lorica trumpet-shaped or bowl-shaped (50-300 μm). Single cell inside lorica. Oral cilia protrude from lorica opening. Lorica may be agglutinated (covered with debris).",
}

# Combined catalog for multi-domain evaluation
def get_catalog(domain):
    """Get the appropriate catalog for a domain."""
    if domain in ['IFCB', 'IFCB-WHOI', 'IFCB-NES']:
        return MARINE_CATALOG
    elif domain in ['ZooLake', 'ZooLake2', 'ZooLake35']:
        return FRESHWATER_CATALOG
    elif domain in ['ZooScan']:
        return MARINE_CATALOG
    else:
        # Return both for unknown domains
        return {**FRESHWATER_CATALOG, **MARINE_CATALOG}


def format_catalog_for_prompt(catalog):
    """Format catalog as text for VLM prompt."""
    lines = []
    for species, description in catalog.items():
        lines.append(f"- {species}: {description}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Freshwater catalog: {len(FRESHWATER_CATALOG)} species")
    print(f"Marine catalog: {len(MARINE_CATALOG)} species")
    print(f"Total unique: {len(set(FRESHWATER_CATALOG.keys()) | set(MARINE_CATALOG.keys()))} species")
