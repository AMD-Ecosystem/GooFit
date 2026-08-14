set(_CORE_SOURCES
    HelpPrinter.cpp
    Variable.cpp
    DataSet.cpp
    BinnedDataSet.cpp
    UnbinnedDataSet.cpp
    PdfBase.cpp
    Version.cpp
    FitControl.cpp
    Application.cpp
    FitManager.cpp)

add_library(_Core STATIC ${_CORE_SOURCES})
goofit_adopt_hip_target(_Core ${_CORE_SOURCES})

target_link_libraries(_Core PRIVATE _goofit_python)
set_target_properties(_Core PROPERTIES FOLDER python)
