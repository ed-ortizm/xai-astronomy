"""Common used fuctions while dealing with spectra"""


def get_anomaly_score_name(
    metric: str, velocity: int, relative: bool, percentage: int
) -> str:
    """Obtain name of the anomaly score naming convention in the project"""

    score_name = f"{metric}"

    if velocity != 0:
        score_name = f"{score_name}_filter_{velocity}"
    if percentage != 100:
        score_name = f"{score_name}_{percentage}"
    if relative is True:
        score_name = f"{score_name}_rel"

    return score_name
