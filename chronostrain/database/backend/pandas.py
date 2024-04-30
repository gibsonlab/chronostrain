from typing import List, Iterator, Dict
import pandas as pd

from chronostrain.model import Marker, Strain
from .base import AbstractStrainDatabaseBackend
from ..error import QueryNotFoundError


class PandasAssistedBackend(AbstractStrainDatabaseBackend):
    def __init__(self):
        self.strains: Dict[str, Strain] = {}
        self.markers: Dict[int, Marker] = {}
        self.strain_df = pd.DataFrame({
            'StrainId': pd.Series(dtype='str'),
            'MarkerIdx': pd.Series(dtype='int')
        }).set_index('MarkerIdx')  # optimize for strains_with_markers() queries.

        self.marker_df = pd.DataFrame({
            'MarkerIdx': pd.Series(dtype='int'),
            'MarkerId': pd.Series(dtype='str'),
            'MarkerName': pd.Series(dtype='str')
        }).set_index(['MarkerId'])  # optimize for get_marker() queries.

    def add_strains(self, strains: Iterator[Strain]):
        strain_df_entries = []
        for strain in strains:
            self.strains[strain.id] = strain
            for marker in strain.markers:
                m_idx = self.add_marker(marker)
                self.markers[m_idx] = marker

                strain_df_entries.append({
                    'StrainId': strain.id,
                    'MarkerIdx': m_idx
                })
        self.strain_df = pd.concat([self.strain_df, pd.DataFrame(strain_df_entries)], ignore_index=True)

    def add_marker(self, marker: Marker) -> int:
        """
        Add a marker and return its id, automatically handle duplicates.
        Pandas-specific implementation.
        """
        try:
            hits = self.marker_df.loc[marker.id, 'MarkerIdx']
            return hits.head(1).item()
        except KeyError:
            new_idx = self.marker_df.shape[0]
            self.marker_df.loc[marker.id] = [
                new_idx, marker.name
            ]
            return new_idx

    def get_strain(self, strain_id: str) -> Strain:
        try:
            return self.strains[strain_id]
        except KeyError:
            raise QueryNotFoundError("Unable to find strain with id `{}`.".format(strain_id))

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(strain_id) for strain_id in strain_ids]

    def _get_marker_index(self, marker_id: str) -> int:
        hits = self.marker_df.loc[marker_id, :]
        if isinstance(hits, pd.Series):
            return hits['MarkerIdx']
        elif isinstance(hits, pd.DataFrame):
            if hits.shape[0] == 0:
                raise RuntimeError(f'Database does not contain marker with ID {marker_id}.')
            elif hits.shape[0] > 1:
                raise RuntimeError(f'Multiple hits found with marker ID {marker_id}.')
            else:
                return hits.head(1)['MarkerIdx'].item()

    def get_marker(self, marker_id: str) -> Marker:
        try:
            return self.markers[self._get_marker_index(marker_id)]
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

    def get_strain_with_marker(self, marker: Marker) -> Strain:
        m_idx = self.marker_df.loc[
            marker.id,
            "MarkerIdx"
        ]

        hit = self.strain_df.loc[m_idx, 'StrainId']
        return self.strains[hit]

    def get_markers_by_name(self, marker_name: str) -> List[Marker]:
        hits = self.marker_df.loc[
            self.marker_df['MarkerName'] == marker_name,
            "MarkerIdx"
        ]
        return [
            self.markers[marker_idx]
            for idx, marker_idx in hits.items()
        ]

    def signature(self) -> str:
        import hashlib
        strain_hash = hashlib.sha256(self.strain_df.to_json().encode()).hexdigest()
        marker_hash = hashlib.sha256(self.marker_df.to_json().encode()).hexdigest()
        return f'PANDAS=<strain:{strain_hash}|marker:{marker_hash}>'
