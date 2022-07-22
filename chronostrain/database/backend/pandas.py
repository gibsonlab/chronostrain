from typing import List, Iterator
import pandas as pd

from chronostrain.model import Marker, Strain
from .base import AbstractStrainDatabaseBackend
from ..error import QueryNotFoundError


class PandasAssistedBackend(AbstractStrainDatabaseBackend):
    def __init__(self):
        self.strains = {}
        self.markers = {}
        self.strain_df = pd.DataFrame({
            'StrainId': pd.Series(dtype='str'),
            'MarkerId': pd.Series(dtype='str')
        })

        self.marker_df = pd.DataFrame({
            'MarkerId': pd.Series(dtype='str'),
            'MarkerName': pd.Series(dtype='str'),
            'IsCanonical': pd.Series(dtype='bool')
        })

    def add_strains(self, strains: Iterator[Strain]):
        strain_df_entries = []
        marker_df_entries = []
        for strain in strains:
            self.strains[strain.id] = strain
            for marker in strain.markers:
                self.markers[marker.id] = marker

                strain_df_entries.append({
                    'StrainId': strain.id,
                    'MarkerId': marker.id
                })

                marker_df_entries.append({
                    'MarkerId': marker.id,
                    'MarkerName': marker.name,
                    'IsCanonical': marker.is_canonical
                })
        self.strain_df = pd.concat([self.strain_df, pd.DataFrame(strain_df_entries)])
        self.marker_df = pd.concat([self.marker_df, pd.DataFrame(marker_df_entries)])

    def get_strain(self, strain_id: str) -> Strain:
        try:
            return self.strains[strain_id]
        except KeyError:
            raise QueryNotFoundError("Unable to find strain with id `{}`.".format(strain_id))

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(strain_id) for strain_id in strain_ids]

    def get_marker(self, marker_id: str) -> Marker:
        try:
            return self.markers[marker_id]
        except KeyError:
            raise QueryNotFoundError("Unable to find marker with id `{}`.".format(marker_id))

    def num_strains(self) -> int:
        return len(self.strains)

    def num_markers(self) -> int:
        return len(self.markers)

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def all_markers(self) -> List[Marker]:
        return list(self.markers.values())

    def get_strains_with_marker(self, marker: Marker) -> List[Strain]:
        hits = self.strain_df.loc[
            self.strain_df['MarkerId'] == marker.id,
            "StrainId"
        ]
        return [
            self.strains[strain_id]
            for idx, strain_id in hits.items()
        ]

    def get_markers_by_name(self, marker_name: str) -> List[Marker]:
        hits = self.marker_df.loc[
            self.marker_df['MarkerName'] == marker_name,
            "MarkerId"
        ]
        return [
            self.markers[marker_id]
            for idx, marker_id in hits.items()
        ]

    def get_canonical_marker(self, marker_name: str) -> Marker:
        hits = self.marker_df.loc[
            (self.marker_df['MarkerName'] == marker_name) & (self.marker_df['IsCanonical']),
            "MarkerId"
        ]

        if hits.shape[0] == 0:
            raise RuntimeError("No canonical markers found with name `{}`.".format(marker_name))

        for idx, marker_id in hits.items():
            return self.get_marker(marker_id)

    def all_canonical_markers(self) -> List[Marker]:
        hits = self.marker_df.loc[
            (self.marker_df['IsCanonical']),
            "MarkerId"
        ]
        return [
            self.markers[marker_id]
            for idx, marker_id in hits.items()
        ]

    def num_canonical_markers(self) -> int:
        return self.marker_df.loc[
            (self.marker_df['IsCanonical']),
            "MarkerId"
        ].shape[0]
