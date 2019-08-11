NEAT Model
==========
This is the model discription.

Genome
------
Text

|genome|

Mutation
--------
Text

|mutation|

Mating
------
Text

|mating|

Speciation
----------
The speciation process has been modified in regard to the source paper.
The new version categorises species by an id which is created using sorted innovation numbers.
A speceis with two genes with innovation 7 and 3 would have the species id = 37.

The original uses a continous scalla to represent speciation.
The distance between two species is calculated as follows:

.. math::
   \delta = \frac{c_1 E}{N} + \frac{c_2 D}{N} + c_3 \cdot{\overline{W}}

Where E is the number of excess nodes D the number if disjoint nodes and W the average weight difference.
N is the maximum number of genes of the two compared nodes.

.. |genome| image:: /_static/genome.png
.. |mating| image:: /_static/mating_neat.png
.. |mutation| image:: /_static/mutation.png