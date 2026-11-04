# samples/multi_stack

Three stacked blocks. Assembly id: `stack`. The folder name and the id differ on purpose: the command uses the id.

```bash
gnn-assemble plan --dir samples --id stack --output stack.3mf
```

`base.obj`, `mid.obj`, and `top.obj` sit on each other, each smaller than the one below. `assembly.json` marks both contacts as prismatic along Z.

The only stable disassembly is `top`, then `mid`, then `base`. Pulling a lower block out first leaves the block above it unsupported. The forward task order is `base`, `mid`, `top`, and each later task lists the block it sits on as a prerequisite.
