# Version 3 BanglaBERT License Decision

## Decision

**Status: PROCEED FOR PRIVATE ACADEMIC DEVELOPMENT — public redistribution remains unresolved.**

The project will proceed in good faith as a noncommercial academic/research use under CC BY-NC-SA 4.0. This is a documented project decision based on the user's factual description, not formal legal advice and not a statement from the licensors.

The user states that this is a university student competition; the work is academic and noncommercial; the model will be used only for the datathon; and there is no company/client use, product development, sale, paid deployment, or commercial service. Fine-tuned weights will remain private unless licensing and organizer requirements for publication are later confirmed. The official rulebook likewise calls the event research-focused and permits publicly available open-weight models and pretrained-model fine-tuning.

## Evidence reviewed

| Resource | Primary evidence | Recorded license or rule | Decision impact |
|---|---|---|---|
| Official BanglaBERT checkpoint | [`csebuetnlp/banglabert` model card](https://huggingface.co/csebuetnlp/banglabert/blob/9ce791f330578f50da6bc52b54205166fb5d1c8c/README.md), pinned immutable revision `9ce791f330578f50da6bc52b54205166fb5d1c8c` | Model-card metadata declares `cc-by-nc-sa-4.0`; the card identifies an ELECTRA discriminator checkpoint and requires its specific normalizer. | Use is limited by Attribution, NonCommercial, and ShareAlike terms. Citation is required but is not a substitute for license compliance. |
| Official Bengali normalizer | [`csebuetnlp/normalizer` repository](https://github.com/csebuetnlp/normalizer) | The official README restricts the repository to non-commercial research under CC BY-NC-SA 4.0. It also documents NFKC as the default Unicode normalization. | Vendoring or attaching the normalizer requires compatible use, attribution, license notice, change notice when applicable, and ShareAlike handling when applicable. |
| CC BY-NC-SA 4.0 terms | [Creative Commons legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) | Defines NonCommercial as not primarily intended for commercial advantage or monetary compensation; grants reproduction/adaptation rights only for NonCommercial purposes; imposes attribution and ShareAlike conditions when licensed or adapted material is shared. | Applying these definitions to a prize competition, private organizer handoff, and fine-tuned ML weights requires a legal or licensor determination that this audit cannot invent. |
| Offline Kaggle mirror | Kaggle API metadata for `reasat/banglabert`, dataset ID `2896818`, retrieved 2026-07-16 | Metadata reports `licenses: [{"name": "unknown"}]`; it provides citations to the BanglaBERT paper and repository but no redistribution or derived-weight terms. | The mirror's authority to redistribute the checkpoint is not established. Citation does not cure the missing mirror license. |
| Competition rules | Repository source of truth: `ICT Fest Datathon Rulebook 2026.md`, especially Models, Training and Compute Rules and Phase 2 requirements | Publicly available open-weight models and fine-tuning are allowed; attached Kaggle/Hugging Face model artifacts are expected for reproducible offline execution. | The rulebook permits the model category but does not interpret CC BY-NC-SA, classify the prize competition as NonCommercial, or specify a compatible license for derived checkpoints. |

No model weights were downloaded or loaded during the license gate itself. In
the later packaging-only authentication phase, the pinned official snapshot was
downloaded, strictly verified, and uploaded to a private Kaggle dataset under
the safeguards below. A subsequent CPU-only synthetic smoke test loaded the
authenticated checkpoint without training, saving derived weights, or using
competition data. The official normalizer was pinned at
`d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9`, minimally vendored with notice,
and parity-tested on synthetic text.

## Use-by-use decision matrix

| Proposed action | Current decision | Reason clearance is incomplete |
|---|---|---|
| Fine-tune BanglaBERT for this university competition | Proceed under the project decision | The user explicitly characterizes the work as academic, student-led, and noncommercial, with no product, client, sale, deployment, or commercial-service use. |
| Store a derived checkpoint as private Kaggle notebook output | Proceed with safeguards | Keep it private, noncommercial, attributed, Git-ignored, and unavailable from public model/dataset hosting. |
| Submit a derived checkpoint to organizers for Phase 2 reruns | Proceed privately with safeguards | Supply it only for academic competition evaluation with CC BY-NC-SA 4.0 notices, attribution, modification notice, and data/model-source notes. |
| Publish derived weights through Kaggle, GitHub, Hugging Face, or another public host | Unresolved and prohibited pending clearance | It is unresolved whether the fine-tuned weights are Adapted Material under applicable law. If they are, ShareAlike and NonCommercial conditions would be relevant; this repository will not guess. |
| Redistribute the `reasat/banglabert` mirror | Not permitted by current evidence | The mirror metadata says `unknown` license and does not establish redistribution authority. |
| Vendor the minimal official normalizer | Proceed with safeguards | Preserve copyright/license/attribution/citation, identify vendored or adapted code, and use it only for this private noncommercial academic workflow. |

## Required attribution and handling safeguards

The implementation must retain and distribute, where applicable:

- identification of the BanglaBERT and normalizer creators;
- the original copyright notices supplied with the materials;
- notice and link for CC BY-NC-SA 4.0;
- links to the original model and normalizer sources;
- the required BanglaBERT and normalizer paper citations;
- an indication that the materials were modified or fine-tuned;
- a compatible ShareAlike license for any output determined to be Adapted Material; and
- no additional terms or technical restrictions that conflict with recipients' licensed rights.

All downloaded base weights, fine-tuned weights, caches, and checkpoints must remain Git-ignored, outside GitHub, private in Kaggle outputs, and unavailable through public Kaggle datasets, Hugging Face repositories, or other public weight hosting. The Kaggle mirror must not be publicly redistributed. The exact application of ShareAlike to fine-tuned weights remains unresolved; no public publication is allowed until the intended publication is confirmed noncommercial and all applicable attribution, ShareAlike, and organizer requirements are satisfied.

## Exact organizer question

> May teams use and fine-tune `csebuetnlp/banglabert` and its official normalizer, both licensed CC BY-NC-SA 4.0 for non-commercial use, in this prize-bearing university datathon; store the derived checkpoint as private Kaggle notebook output; provide it to organizers for offline Phase 2 reruns; and, if required, publish it under CC BY-NC-SA 4.0 with full attribution? Please confirm whether this complies with the competition rules, whether public weight publication is required, and whether the organizers have permission from the model licensors for these uses.

This question may be sent in parallel. Written organizer permission is no longer a hard stop for private development and competition participation under the user's clarified noncommercial academic-use facts. Organizer confirmation still matters before any public checkpoint publication or if competition rules change.

## Remaining hard stops

Version 3 must stop if any of the following occurs:

1. mirror files cannot be authenticated against a pinned immutable official revision;
2. required mirror and official hashes differ unexpectedly;
3. required tokenizer or model files are absent;
4. executable offline loading fails or ELECTRA encoder parameters are missing;
5. the official normalizer cannot be used with the required attribution;
6. competition rules explicitly prohibit this model or handling arrangement; or
7. the solution requires public weight redistribution under terms that have not been confirmed compatible.

The mirror's `unknown` Kaggle metadata license does not override the official model license. It does require complete file-by-file authentication against the pinned official snapshot before any mirror weight is loaded. Public derived-weight publication remains unresolved and prohibited; private development may proceed through the remaining technical gates.
