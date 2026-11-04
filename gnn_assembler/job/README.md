# gnn_assembler/job

Serializes the forward assembly into the job file a manufacturing tool can open.

| File | What it does |
| --- | --- |
| `tasks_xml.py` | Builds one `<Task>` per part in insertion order. The task stores the seated position, the insertion axis and angle (the extraction negated), the G-code path, and `<Prerequisites>` for earlier tasks whose parts it touches. The first part has an empty list. |
| `threemf.py` | Builds the `.3mf` zip in memory with deflated entries. `3D/3dmodel.model` holds the part meshes. `Attachments/Tasks.xml` and the G-code files sit beside it. `[Content_Types].xml`, `_rels/.rels`, and `3D/_rels/3dmodel.model.rels` bind those attachments to the model. |
