# samples/rotational

Two-part bayonet. Assembly id: `rotational`.

`shaft.obj` is a vertical post, a wide flange at the bottom, and a tooth on the side. `collar.obj` is a square sleeve with a notch. The tooth sits above the collar, so a pure vertical lift hits it. After a quarter turn the notch lines up with the tooth and the collar can rise. There is no joint override in `assembly.json`. The contact graph labels the pair revolute because the collar wraps the shaft and cannot translate off it.

The forward assembly places the shaft, then turns and lowers the collar.
