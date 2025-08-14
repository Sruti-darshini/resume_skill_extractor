# launch.spec
block_cipher = None

a = Analysis(
    ['launch.py', 'app_desktop.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates/*', 'templates'),
        ('fine_tuned_skill_model/*', 'fine_tuned_skill_model'),
        ('fine_tuned_ner_model/*', 'fine_tuned_ner_model'),
        ('app_desktop.py', '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,   # include binaries inside the exe
    a.datas,      # include data inside the exe
    [],
    name='MyPyQt5App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,  # run from temp dir, no _internal in dist
    console=False,        # set to True if you want console output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
