generate_unit(static_transform_publisher)

# TODO: query the yaml for the types
target_link_libraries(unit_static_transform_publisher tf2_basis)
target_compile_definitions(unit_static_transform_publisher PUBLIC "BASIS_PLATFORM=${BASIS_PLATFORM}")
