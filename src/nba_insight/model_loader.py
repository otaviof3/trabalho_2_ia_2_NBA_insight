
import joblib

from .config import MODEL_PATH


def load_model_artifact(path=MODEL_PATH):
    """
    Carrega os componentes do modelo K-Means salvos em disco.

    O arquivo deve conter:
    - modelo: K-Means treinado
    - imputer: SimpleImputer treinado
    - scaler: StandardScaler treinado
    - features: lista de atributos usados no treinamento
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Artefato do modelo não encontrado: {path}\n"
            "Execute o script de treinamento/exportação do modelo "
            "antes de utilizar esta função."
        )

    artifact = joblib.load(path)

    required_keys = [
        "modelo",
        "imputer",
        "scaler",
        "features",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing_keys:
        raise ValueError(
            "O artefato do modelo não possui os componentes necessários: "
            + ", ".join(missing_keys)
        )

    return artifact


def get_model():
    """
    Retorna o K-Means treinado.
    """

    artifact = load_model_artifact()

    return artifact["modelo"]


def get_preprocessing_components():
    """
    Retorna o imputer, scaler e lista de features.
    """

    artifact = load_model_artifact()

    return (
        artifact["imputer"],
        artifact["scaler"],
        artifact["features"],
    )
