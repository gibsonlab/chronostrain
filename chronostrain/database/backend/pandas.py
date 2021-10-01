from typing import List
import pandas as pd

from chronostrain.model import Marker, Strain
from .base import AbstractStrainDatabaseBackend, QueryNotFoundError


class PandasAssistedBackend(AbstractStrainDatabaseBackend):
    def __init__(self):
        self.strains = {}
        self.markers = {}
        self.strain_df = pd.DataFrame({
            'Strain': pd.Series(dtype='str'),
            'Marker': pd.Series(dtype='str')
        })

    def add_strain(self, strain: Strain):
        self.strains[strain.id] = strain
        for marker in strain.markers:
            self.markers[marker.id] = marker
            self.strain_df.append({
                'Strain': strain.id,
                'Marker': marker.id
            })

    def get_strain(self, strain_id: str) -> Strain:
        try:
            return self.strains[strain_id]
        except KeyError:
            raise QueryNotFoundError(strain_id)

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(strain_id) for strain_id in strain_ids]

    def get_marker(self, marker_id: str) -> Marker:
        try:
            return self.markers[marker_id]
        except KeyError:
            raise QueryNotFoundError(marker_id)

    def num_strains(self) -> int:
        return len(self.strains)

    def num_markers(self) -> int:
        return len(self.markers)

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def all_markers(self) -> List[Marker]:
        return list(self.markers.values())

    def get_strains_with_marker(self, marker_name: str) -> List[Strain]:
        hits = self.strain_df.loc[
            self.strain_df['Marker'] == marker_name,
            "Strain"
        ]
        if len(hits) == 0:
            raise QueryNotFoundError(marker_name)
        return [
            self.strains[strain_id]
            for idx, strain_id in hits.items()
        ]
