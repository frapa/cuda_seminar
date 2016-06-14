#version 330

uniform sampler2D tex;
in vec2 texCoord;
out vec4 colorOut;

void main() {
	float strength = texture(tex, texCoord).x;

	//	Colormap
	//		1 		red			(1, 0, 0)
	//		0.75 	yellow		(1, 1, 0)
	//		0.5 	green		(0, 1, 0)
	//		0.25	cyan		(0, 1, 1)
	//		0   	blue        (0, 0, 1)
	// RED
	float red = (strength > 0.75) ?
		1 : ((strength > 0.5) ?
			(strength - 0.5) * 4 : 0);
	// GREEN
	float green = (strength <= 0.75 && strength > 0.25) ?
		1 : ((strength > 0.75) ? 1 - strength : strength) * 4;
	// BLUE
	float blue = (strength > 0.5) ?
		0 : ((strength > 0.25) ?
			(0.5 - strength) * 4 : 1);

    colorOut = vec4(red, green, blue, 1);
}
