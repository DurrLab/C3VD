/*  Adapted from Model.h in OWL repository by Ingo Wald.
    https://github.com/owl-project/
*/

#ifndef MODEL_H_
#define MODEL_H_

#include <owl/owl.h>
#include <owl/common/math/AffineSpace.h>
#include <vector>
  
/*! a simple indexed triangle mesh that our sample renderer will
    render */
struct TriangleMesh {
    std::vector<owl::vec3f> vertex;
    std::vector<owl::vec3f> normal;
    std::vector<owl::vec2f> texcoord;
    std::vector<owl::vec3i> index;

    // material data:
    owl::vec3f              diffuse;
    int                     diffuseTextureID { -1 };
};

struct Texture {
    ~Texture()
    { if (pixel) delete[] pixel; }

    uint32_t *pixel      { nullptr };
    owl::vec2i     resolution { -1 };
};

struct Model {
    ~Model()
    {
        for (auto mesh : meshes) delete mesh;
        for (auto texture : textures) delete texture;
    }

    std::vector<TriangleMesh *> meshes;
    std::vector<Texture *>      textures;

    //! bounding box of all vertices in the model
    owl::box3f bounds;
};

Model *loadOBJ(const std::string &objFile);

#endif /* MODEL_H_ */