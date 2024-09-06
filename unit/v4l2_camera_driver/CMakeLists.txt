generate_unit(v4l2_camera_driver)

# TODO: query the yaml for the types
target_link_libraries(unit_v4l2_camera_driver image_conversion foxglove_schemas_protobuf)
