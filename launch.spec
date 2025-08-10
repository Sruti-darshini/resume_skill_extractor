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
    [],
    exclude_binaries=True,
    name='MyPyQt5App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False  # True if you want console for debugging
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MyPyQt5App'
)
