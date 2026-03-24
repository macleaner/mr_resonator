# ---- material defaults (single source of truth) ----
# Keep these as the *base* values you think are most intuitive to edit.
# (You can add/remove keys freely.)
E_CHARGE = 1.602176634e-19   # J/eV
kB = 1.380649e-23            # J/K

MATERIAL_DEFAULTS = {
    "Al": {
        "N0": 1.72e10 / E_CHARGE,   # um^-3 J^-1 Gao thesis
        "tau0": 438e-9,             # s (Kaplan 1976)
        "Tc": 1.2,                  # K
        "Rsheet_N": 0.66,           # ohm/sq
    },
    "TiN_sub_superspec": {
        "N0": 3.9e10 / E_CHARGE, # Gao 2012 ***
        "tau0": 88e-9, # Kardakova 2013
        "Tc": 2.0, 
        "rhoN": 170e-6 / 100, # uohm * cm / 100 = uohm*m - Pete's thesis (for 20nm film at Tc 2 K) -- this is anomalously low
    },
    "TiN_stoich_leduc": { # vissers 2012 gives rhoN between 45 - 185 uOhm cm; driessen 2012 up to 380u; generally lower for higher Tc
        "N0" : 8.7e9 / E_CHARGE, # LeDuc 2012 *** notes that this is insensitive to stoichiometry
        "tau0": 88e-9, # Kardakova 2013
        "Tc": 4.1, # Leduc 2012
        "rhoN": 100e-6 /100 # uohm * cm / 100 = uohm*m - Leduc 2012 (20 to 100nm films, Tc 0.7 to 4.1 K; Ls=8.4 pH)
    },
    "TiN_sub_leduc": {
        "N0" : 8.7e9 / E_CHARGE, # LeDuc 2012 *** notes that this is insensitive to stoichiometry
        "tau0": 88e-9, # Kardakova 2013
        "Tc": 1.1, # Leduc 2012
        "rhoN": 100e-6 /100 # uohm * cm / 100 = uohm*m - Leduc 2012 (20 to 100nm films, Tc 0.7 to 4.1 K; Ls=8.4 pH)
    },
    "NbTiN": {
        "N0": 1.0e10 / E_CHARGE,
        "tau0": 10e-9,
        "Tc": 16.0,
        "Rsheet_N": 45.0,
    },
    "NbN": {
        "N0": 1.15e10 / E_CHARGE, # Barends 2011
        # "tau0": 10e-9, # approx., incl phonon trapping (Semenov 1997)
        "tau0": 30e-12, # Semenov 
        "Tc": 16.0,
        "Rsheet_N": 70.0,
    },
    "Nb": {
        "N0": 3.17e10 / E_CHARGE, # (Kaplan 1976)
        "tau0": 0.149e-9, # (Kaplan 1976)
        "Tc": 9.0,
        "Rsheet_N": 5.0,
    },
}


def resolve_material_properties(material, thickness, overrides=None):
    """
    Returns a dict containing the resolved material properties.

    Rules:
      1) Start from MATERIAL_DEFAULTS[material]
      2) Apply overrides (if any) so user-specified values win
      3) Compute derived quantities (rhoN, sigmaN, Delta)

    Supported "normal-state" override styles:
      - specify Rsheet_N (ohm/sq)
      - OR specify rhoN (ohm*m)
      - OR specify sigmaN (S/m)
    Precedence if multiple are provided:
      sigmaN > rhoN > Rsheet_N
    """
    if material not in MATERIAL_DEFAULTS:
        raise ValueError(f"Unknown material {material!r}. Options: {list(MATERIAL_DEFAULTS)}")

    props = MATERIAL_DEFAULTS[material].copy()

    if overrides:
        # only apply non-None overrides
        for k, v in overrides.items():
            if v is not None:
                props[k] = v

    # --- normal-state conductivity/resistivity bookkeeping ---
    if "sigmaN" in props:
        props["rhoN"] = 1.0 / props["sigmaN"]
        props["Rsheet_N"] = props["rhoN"] / thickness
    elif "rhoN" in props:
        props["sigmaN"] = 1.0 / props["rhoN"]
        props["Rsheet_N"] = props["rhoN"] / thickness
    else:
        props["rhoN"] = props["Rsheet_N"] * thickness
        props["sigmaN"] = 1.0 / props["rhoN"]

    # --- superconducting gap ---
    props["Delta"] = 1.76 * kB * props["Tc"]

    # bookkeeping (optional)
    props["material"] = material
    props["thickness"] = thickness

    return props
