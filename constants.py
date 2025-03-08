from enum import Enum


class SuccessFailure(Enum):
    SUCCESS = 1
    FAILURE = 0


class Sex(Enum):
    MALE = 1
    FEMALE = 2


class PracticeLevel(Enum):
    AGEN = 0
    FGP = 1


class SealerType(Enum):
    AH_PLUS = 1
    BC_SEALER = 2
    RIBBON = 3
    KERR = 4
    NEOSEALER_FLO = 5
    ROTH = 6


class RootFillingDensity(Enum):
    NON_FILLED = 0
    ADEQUATE = 1    # No Voids
    INADEQUATE = 2  # Has Voids


class RootFillingLength(Enum):
    NON_FILLED = 0
    ADEQUATE = 1      # 0-2mm from radiographic apex
    NON_ADEQUATE = 2  # >2mm short of radiographic apex
    OVERFILLED = 3    # Past radiographic apex


class PreTxPulpalDiagnosis(Enum):
    NORMAL_PULP = 0
    REVERSIBLE_PULPITIS = 1
    SYMPTOMATIC_IRREVERSIBLE_PULPITIS = 2
    ASYMPTOMATIC_IRREVERSIBLE_PULPITIS = 3
    NECROTIC_PULP = 4


class PreTxPeriapicalDiagnosis(Enum):
    NORMAL_APICAL_TISSUES = 0
    SYMPTOMATIC_APICAL_PERIODONTITIS = 1
    ASYMPTOMATIC_APICAL_PERIODONTITIS = 2
    CHRONIC_APICAL_ABSCESS = 3
    ACUTE_APICAL_ABSCESS = 4


class ApicalExtensionOfPost(Enum):
    ABSENT = 0    # No Post
    LT_6_MM = 1   # <6mm
    GTE_6_MM = 2  # >=6mm


class ToothType(Enum):
    MAXILLARY_ANTERIOR_TOOTH = 1
    MANDIBULAR_ANTERIOR_TOOTH = 2
    MAXILLARY_PREMOLAR_TOOTH = 3
    MANDIBULAR_PREMOLAR_TOOTH = 4
    MAXILLARY_MOLAR_TOOTH = 5
    MANDIBULAR_MOLAR_TOOTH = 6


TOOTH_TO_TOOTH_TYPE_MAPPING = {
    ToothType.MAXILLARY_ANTERIOR_TOOTH: [6, 7, 8, 9, 10, 11],
    ToothType.MANDIBULAR_ANTERIOR_TOOTH: [22, 23, 24, 25, 26],
    ToothType.MAXILLARY_PREMOLAR_TOOTH: [4, 5, 12, 13],
    ToothType.MANDIBULAR_PREMOLAR_TOOTH: [20, 21, 28, 29],
    ToothType.MAXILLARY_MOLAR_TOOTH: [1, 2, 3, 14, 15, 16],
    ToothType.MANDIBULAR_MOLAR_TOOTH: [17, 18, 19, 30, 31, 32]
}
