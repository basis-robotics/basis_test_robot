include(FindProtobuf)
find_package(Protobuf REQUIRED)

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")


function(declare_proto proto_name)

  add_library(${proto_name}_proto SHARED 
    ${proto_name}.proto
    )
  target_link_libraries(${proto_name}_proto PUBLIC basis::plugins::serialization::protobuf)

  set(BASIS_GENERATED_DIR "${GENERATED_DIR}/${proto_name}_proto")

  make_directory(${BASIS_GENERATED_DIR})
  target_include_directories(${proto_name}_proto SYSTEM PUBLIC "$<BUILD_INTERFACE:${BASIS_GENERATED_DIR}>")

  protobuf_generate(
    LANGUAGE cpp
    TARGET ${proto_name}_proto 
    PROTOC_OUT_DIR "${BASIS_GENERATED_DIR}"
    )

  install(TARGETS ${proto_name}_proto DESTINATION lib/)
endfunction()

declare_proto(basis_robot_input)
declare_proto(basis_robot_state)