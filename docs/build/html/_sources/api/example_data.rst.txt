Example Datasets
================

torch_openreml.example_data.john_alpha
--------------------------------------

.. py:data:: torch_openreml.example_data.john_alpha

Alpha lattice design of spring oats (re-exported from R ``agridat``).

This dataset is a Python re-export of ``john.alpha`` from the R package
``agridat``. It contains field trial data from a resolvable alpha lattice
design conducted at Craibstone near Aberdeen.

The experiment includes 24 oat varieties evaluated across 3 replicates,
with each replicate arranged into 6 incomplete blocks of size 4.

**Shape:** 72 rows × 7 columns

Example
~~~~~~~

.. jupyter-execute::

    from torch_openreml.example_data import john_alpha
    print(john_alpha)

Columns
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 50

   * - Column
     - Type
     - Description
   * - ``plot``
     - int
     - Plot number.
   * - ``rep``
     - int
     - Replicate identifier (1–3).
   * - ``block``
     - int
     - Incomplete block within replicate.
   * - ``gen``
     - str
     - Genotype / variety identifier.
   * - ``yield``
     - float
     - Dry matter yield (tonnes/ha).
   * - ``row``
     - int
     - Field row coordinate (note: layout is linear, not grid-based).
   * - ``col``
     - int
     - Field column coordinate.

Design
~~~~~~

Resolvable alpha lattice design with incomplete blocking structure.

.. note::

    The tabular layout in John & Williams (1995, p. 146) does not represent
    the physical field layout (plots were arranged in a single line).

Source
~~~~~~

Re-exported from R package ``agridat`` (``john.alpha`` dataset).

References
~~~~~~~~~~

J. A. John & E. R. Williams (1995). Cyclic and computer generated designs. Chapman and Hall, London. Page 146.