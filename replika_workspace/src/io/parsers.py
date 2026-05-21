from dataclasses import dataclass
from typing import BinaryIO, Tuple, NamedTuple, Optional
import numpy as np


@dataclass(frozen=True)
class VTKHeader:
    """
    Metadata parsed from the ASCII header of a VTK file (Version 2.0).

    Attributes:
        version: The VTK version string (e.g., "2.0").
        title: The descriptive title provided in the header.
        encoding: Data format (BINARY or ASCII).
        dataset_type: The VTK dataset type (e.g., STRUCTURED_POINTS).
        n_cells: Total number of cells calculated from dimensions.
    """
    version: str
    title: str
    encoding: str
    dataset_type: str
    n_cells: int


class VTKHeaderParser:
    """
    Responsible for extracting header metadata and positioning the file pointer
    at the start of the binary data segment.
    """

    def parse(self, stream: BinaryIO) -> VTKHeader:
        """
        Reads the ASCII portion of the VTK file to populate a VTKHeader.

        Args:
            stream: A binary stream opened at the start of the VTK file.

        Returns:
            VTKHeader: The parsed metadata object.

        Raises:
            ValueError: If the file does not match expected VTK 2.0 format
                or dataset types.
        """
        try:
            # 1. Version line: "# vtk DataFile Version 2.0"
            line = stream.readline()
            if not line:
                raise ValueError("Empty VTK file stream.")
            line_str = line.decode('ascii').strip()
            if not line_str.startswith('# vtk DataFile Version'):
                raise ValueError(f"Invalid VTK header line 1: {line_str}")
            version = line_str.split('Version ')[-1]
            if version != '2.0':
                raise ValueError(f"Unsupported VTK version: {version}. Expected 2.0")

            # 2. Title line: typically descriptive text
            line = stream.readline()
            if not line:
                raise ValueError("Incomplete VTK header: missing title line.")
            title = line.decode('ascii', errors='replace').strip()

            # 3. Encoding line: BINARY or ASCII
            line = stream.readline()
            if not line:
                raise ValueError("Incomplete VTK header: missing encoding line.")
            encoding = line.decode('ascii').strip()
            if encoding != 'BINARY':
                raise ValueError(f"Unsupported VTK encoding: {encoding}. Expected BINARY.")

            # 4. Dataset type line: e.g. "DATASET STRUCTURED_POINTS"
            line = stream.readline()
            if not line:
                raise ValueError("Incomplete VTK header: missing dataset type line.")
            line_str = line.decode('ascii').strip()
            if not line_str.startswith('DATASET'):
                raise ValueError(f"Invalid VTK header line 4: {line_str}")
            dataset_type = line_str.split()[-1]
            if dataset_type != 'STRUCTURED_POINTS':
                raise ValueError(f"Unsupported VTK dataset type: {dataset_type}. Expected STRUCTURED_POINTS")

            # 5. Search for dimensions and cell count
            nx, ny, nz = 0, 0, 0
            n_cells = 0
            while True:
                pos = stream.tell()
                line_bytes = stream.readline()
                if not line_bytes:
                    break
                
                try:
                    # ASCII portion of the header
                    line_str = line_bytes.decode('ascii').strip()
                except UnicodeDecodeError:
                    # Reached start of binary data prematurely; back up
                    stream.seek(pos)
                    break
                
                if not line_str:
                    continue
                
                parts = line_str.split()
                if not parts:
                    continue
                    
                keyword = parts[0].upper()
                if keyword == 'DIMENSIONS':
                    if len(parts) >= 4:
                        nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
                elif keyword == 'CELL_DATA':
                    n_cells = int(parts[1])
                    # Stop here to allow field parsers to read SCALARS/VECTORS/etc.
                    break
                elif keyword == 'POINT_DATA':
                    # Stop if point data is encountered; n_cells can be derived from DIMENSIONS
                    break
                elif keyword in ('ORIGIN', 'SPACING', 'FIELD'):
                    continue
                elif keyword in ('SCALARS', 'VECTORS', 'TENSORS'):
                    # Data started without explicit CELL_DATA line
                    stream.seek(pos)
                    break
            
            # Use dimensions to calculate cell count if CELL_DATA line was absent
            if n_cells == 0 and nx > 0:
                n_cells = (nx - 1) * (ny - 1) * (nz - 1)
                
            if n_cells == 0:
                raise ValueError("Could not determine cell count (n_cells) from VTK header.")

            return VTKHeader(
                version=version,
                title=title,
                encoding=encoding,
                dataset_type=dataset_type,
                n_cells=n_cells
            )
            
        except (UnicodeDecodeError, IndexError, ValueError) as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"VTK header parsing failed: {str(e)}")
class StructuredPointLayout(NamedTuple):
    """Spatial configuration for STRUCTURED_POINTS datasets."""
    dimensions: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]
class VTKStructuredPointExtractor:
    """
    Extracts the grid layout and verifies data consistency for 
    structured point datasets found in DS1.
    """

    def extract_layout(self, stream: BinaryIO) -> StructuredPointLayout:
        """
        Extracts DIMENSIONS, ORIGIN, and SPACING from the VTK stream.

        Args:
            stream: Binary stream positioned at the start of the description block.

        Returns:
            StructuredPointLayout: The extracted grid parameters.

        Raises:
            ValueError: If required keywords are missing or dimensions are invalid.
        """
        dimensions = None
        origin = None
        spacing = None

        # Keep current position to restore it if needed, or just scan from here.
        # The docstring says stream is positioned at the start of description block.
        # But for robustness, let's just search from the current position.
        
        while not (dimensions and origin and spacing):
            pos = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                line_str = line.decode('ascii').strip()
            except UnicodeDecodeError:
                stream.seek(pos)
                break
            
            if not line_str:
                continue
            
            parts = line_str.split()
            if not parts:
                continue
                
            keyword = parts[0].upper()
            if keyword == 'DIMENSIONS':
                dimensions = (int(parts[1]), int(parts[2]), int(parts[3]))
            elif keyword == 'ORIGIN':
                origin = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif keyword == 'SPACING':
                spacing = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif keyword in ('POINT_DATA', 'CELL_DATA'):
                stream.seek(pos)
                break
        
        if dimensions is None:
            raise ValueError("VTK header missing DIMENSIONS")
        if origin is None:
            origin = (0.0, 0.0, 0.0)
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
            
        return StructuredPointLayout(dimensions, origin, spacing)

    def locate_field_block(self, stream: BinaryIO, field_name: str) -> int:
        """
        Scans the stream for the location of a specific data attribute (e.g., 'dens').

        Args:
            stream: Binary stream to scan.
            field_name: The case-sensitive name of the field (SCALARS or VECTORS).

        Returns:
            int: The byte offset to the start of the binary data for that field.
        
        Raises:
            KeyError: If the field_name is not found in the VTK file.
        """
        stream.seek(0)
        current_count = 0
        
        while True:
            pos = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                line_str = line.decode('ascii').strip()
            except UnicodeDecodeError:
                # We accidentally hit binary data. This shouldn't happen if we skip correctly.
                # But if it does, we might have miscalculated a skip size or reached end of ASCII.
                continue
            
            if not line_str:
                continue
                
            parts = line_str.split()
            keyword = parts[0].upper()
            
            if keyword == 'DIMENSIONS':
                nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
                if current_count == 0:
                    # Default to points count if CELL_DATA not yet seen
                    current_count = nx * ny * nz
            elif keyword == 'POINT_DATA':
                current_count = int(parts[1])
            elif keyword == 'CELL_DATA':
                current_count = int(parts[1])
            elif keyword == 'SCALARS':
                name = parts[1]
                data_type = parts[2].lower()
                num_comp = int(parts[3]) if len(parts) > 3 else 1
                
                # Consume LOOKUP_TABLE line
                lookup_line = stream.readline().decode('ascii').strip()
                # (VTK spec says LOOKUP_TABLE is required)
                
                offset = stream.tell()
                if name == field_name:
                    return offset
                
                # Skip binary data
                item_size = 8 if data_type == 'double' else 4
                stream.seek(offset + current_count * num_comp * item_size)
            elif keyword == 'VECTORS':
                name = parts[1]
                data_type = parts[2].lower()
                
                offset = stream.tell()
                if name == field_name:
                    return offset
                
                # Skip binary data (VECTORS always has 3 components)
                item_size = 8 if data_type == 'double' else 4
                stream.seek(offset + current_count * 3 * item_size)
        
        raise KeyError(f"Field '{field_name}' not found in VTK file.")
class VTKFieldExtractor:
    """
    Handles the extraction of binary multi-variate or scalar data arrays 
    from VTK files, specifically for DS1 (Turbulence Velocity Snapshots).
    """

    def extract_scalar_field(self, stream: BinaryIO, offset: int, count: int) -> np.ndarray:
        """
        Extracts a scalar field (e.g., 'dens') from binary data.

        Args:
            stream: Binary stream positioned at the data offset.
            offset: Byte offset where data begins.
            count: Total number of elements to read (typically prod(dimensions)).

        Returns:
            np.ndarray: A 1D flattened array of scalar values (float32, Big-Endian).

        Raises:
            EOFError: If the stream ends before all values are read.
        """
        stream.seek(offset)
        # Using big-endian float32 as specified
        dtype = np.dtype('>f4')
        raw_data = stream.read(count * dtype.itemsize)
        if len(raw_data) < count * dtype.itemsize:
            raise EOFError(f"Expected {count * dtype.itemsize} bytes, got {len(raw_data)}")
        return np.frombuffer(raw_data, dtype=dtype).copy()

    def extract_vector_field(self, stream: BinaryIO, offset: int, count: int) -> np.ndarray:
        """
        Extracts a 3-component vector field (e.g., velocity 'v') from binary data.

        Args:
            stream: Binary stream positioned at the data offset.
            offset: Byte offset where data begins.
            count: Total number of vectors to read.

        Returns:
            np.ndarray: A 2D array of shape (count, 3) representing (vx, vy, vz).

        Raises:
            EOFError: If the stream ends before all components are read.
        """
        stream.seek(offset)
        # Using big-endian float32 as specified
        dtype = np.dtype('>f4')
        total_elements = count * 3
        raw_data = stream.read(total_elements * dtype.itemsize)
        if len(raw_data) < total_elements * dtype.itemsize:
            raise EOFError(f"Expected {total_elements * dtype.itemsize} bytes, got {len(raw_data)}")
        return np.frombuffer(raw_data, dtype=dtype).reshape((count, 3)).copy()
