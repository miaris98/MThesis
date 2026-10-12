# Moved to external storage

'weights\wor_nc.pth' was moved to E:\MThesis_EXP\checkpoints\local_pretrained\wor_nc.pth
on 2026-09-09 18:02. download_pretrained_weights() in
src\models\world_on_rails\wor_loader.py defaults to save_dir="weights" (relative
to cwd) - if called locally again with no override, it will just re-download into
a fresh ./weights/, since this is a redownloadable pretrained checkpoint, not a
unique project artifact.
