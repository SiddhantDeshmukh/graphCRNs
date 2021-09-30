# TODO list

## Week of 27/09/21

### Solver

* Fix 'rate_dict' feeding into Jacobian to avoid double-counting
* Replace string Jacobian method in 'dynamics.py' with functional Jacobian
- Run test cases across temperature regime with functional Jacobian
  - Potentially check if function can be compiled to improve runtime

### Network

- Finalise solar CO network
  - Add C2, N2, O2, CN, NO, NH to try to stabilise high-temperature behaviour
- Finalise metal-poor CNO network
  - Create more complex network with O2, C2, N2

### Pathfinding

- Fix pathfinding logic (mathematical logic seems sound!)
- Run pathfinding for _final_ solar CO network in characteristic
  regions of the solar photosphere (pick a few)

### Writeup

- Graph CRN theory
- Graph theoretical kinetics
- Examples using ring reaction network and an astrophysical case
  - Perhaps a simplified CO network to illustrate the idea of complexes?
    - Could be hydrogen ionisation as well, plus H2
