#ifndef VAI_COMMON_CONV_BATCH_H
#define VAI_COMMON_CONV_BATCH_H

// A batched convolution stacks its images on the dispatch z axis, the image
// outermost: z = image * zPerImage + z within the image. A kernel reads its
// image once and offsets every per-image buffer by that image's slice.
uint convImage(uint batch)
{
    return gl_WorkGroupID.z / (gl_NumWorkGroups.z / batch);
}

uint convImageZ(uint batch)
{
    return gl_WorkGroupID.z % (gl_NumWorkGroups.z / batch);
}

#endif // VAI_COMMON_CONV_BATCH_H
