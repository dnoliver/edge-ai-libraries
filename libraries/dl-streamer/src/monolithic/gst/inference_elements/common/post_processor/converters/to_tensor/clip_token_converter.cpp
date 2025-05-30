/*******************************************************************************
 * Copyright (C) 2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "clip_token_converter.h"
#include "copy_blob_to_gststruct.h"
#include "inference_backend/image_inference.h"
#include "inference_backend/logger.h"
#include "safe_arithmetic.hpp"

#include <exception>
#include <string>
#include <vector>

using namespace post_processing;
using namespace InferenceBackend;

TensorsTable CLIPTokenConverter::convert(const OutputBlobs &output_blobs) {
    ITT_TASK(__FUNCTION__);
    TensorsTable tensors_table;
    try {
        const size_t batch_size = getModelInputImageInfo().batch_size;
        tensors_table.resize(batch_size);

        // Process only the first blob
        auto blob_iter = output_blobs.begin();
        if (blob_iter != output_blobs.end()) {
            OutputBlob::Ptr blob = blob_iter->second;
            if (not blob) {
                throw std::invalid_argument("Output blob is empty");
            }

            const std::string &layer_name = blob_iter->first; // inference_layer_name:pooler_output:1495

            for (size_t frame_index = 0; frame_index < batch_size; ++frame_index) {
                GstStructure *tensor_data = BlobToTensorConverter::createTensor().gst_structure();

                auto dims = blob->GetDims();
                auto clip_token_size = dims[2]; // tensor shape: [N, 257, 1024]

                CopyOutputBlobToGstStructure(blob, tensor_data, BlobToMetaConverter::getModelName().c_str(),
                                             layer_name.c_str(), batch_size, frame_index,
                                             clip_token_size * sizeof(float));

                // Adjusting tensor dimensions - we retrieved just the token, and skipped the rest of embeddings
                const GValue *dims_value = gst_structure_get_value(tensor_data, "dims");
                GValue *element = (GValue *)gst_value_array_get_value(dims_value, 1);
                g_value_set_uint(element, 1);
                // Now set the modified array back to the structure
                gst_structure_set_value(tensor_data, "dims", dims_value);

                // In different versions of GStreamer, tensors_batch are attached to the buffer in a different order.
                // Thus, we identify our meta using tensor_id.
                gst_structure_set(tensor_data, "tensor_id", G_TYPE_INT, safe_convert<int>(frame_index), NULL);

                std::vector<GstStructure *> tensors{tensor_data};
                tensors_table[frame_index].push_back(tensors);
            }
        }
    } catch (const std::exception &e) {
        GVA_ERROR("An error occurred while processing output BLOBs: %s", e.what());
    }
    return tensors_table;
}