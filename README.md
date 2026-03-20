### Installation

```
git clone git@github.com:aurelio-spadotto/poly-memb.git
cd poly-memb
pip install .
```


### Import

```
from polymemb import mesh_manager as mema # mesh management tools
from polymemb.solvers import ddr          # ddr solver tools
from polymemb.solvers import hho          # hho solver tools
```

### Notes on future developments
poly-memb would like to become a 2D hybrid solver that can be used to produce small scale proofs of concepts for academic/didactic purpose.

In particular, it aims at testing solvers that are based on physical manipulations of the mesh. For example, for clipping with surfaces and non standard refinement techniques. The implementation of these geomtric algorithms can benefit from easy manipulability/visualization, which can be handed nicely within jupyter notebooks.

The aim is to provide a platform that can be simply integrated in jupyter notebooks in order to provide insightful visualization, rapid testing and a suitable format to present the code as a part of an article.

polymemb cannot aim at being a competitive, large scale solver, because of the intrinsic constraint linked to working with an interpreted language. Moreover, it would be redundant, given the availability of performing, optimized solvers that can implement parallelization.

Nevertheless, polymemb should at least aim at getting as performing as a standard c++ implementation on small scale problems, as it is not the case currently.

Goals: assembly of linear systems should be done in less than one second. Implement just-in-time compilation of reepated routines, optimize the assembly loops
   
