# samples/peg_hole

Two-part peg-in-pocket. Assembly id: `peg_hole`.

`base.obj` is the floor plus four walls around a square pocket. `peg.obj` sits on the floor inside that pocket, with a gap to the walls, and sticks out above them. `assembly.json` records both collision boxes and a prismatic joint along Z.

A straight lift of the peg is free. Sliding it into a wall is not. The base stays on the ground, so the peg is removed first. The forward assembly places the base, then lowers the peg.
