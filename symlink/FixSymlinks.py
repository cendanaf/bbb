import os

def FixSymlinks(filePath, rootfsPath='~/BBB/rootfs'):
	# Expand the ~ in rootfs path
	rootfsPath = os.path.expanduser(rootfsPath)
	# Read list of broken symlinks from the text filePath
	with open(filePath, 'r') as file:
		brokenSymlinks = []
		for line in file:
			brokenSymlinks.append(line.strip())

	# Fix each symlink
	for symlinkInfo in brokenSymlinks:
		try:
			# Split line into symlink and target
			symlink, target = symlinkInfo.split(' -> ')
			target = os.path.expanduser(target)

			# Prepend ~/BBB/rootfs/ to the target
			fixedTarget = rootfsPath + target

			# Update the symlink with the fixed target
			os.unlink(symlink)
			os.symlink(fixedTarget, symlink)
			print(f"Fixed symlink: {symlink} -> {fixedTarget}")

		except Exception as e:
			print(f"Error fixing symlink {symlink}: {str(e)}")


if __name__ == "__main__":
	FixSymlinks('broken_symlinks_list.txt')
